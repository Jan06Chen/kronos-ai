from __future__ import annotations

import argparse
import logging
import uuid
from dataclasses import replace
from datetime import date, timedelta

from kronos_a_share_predictor.backtest.engine import BacktestEngine
from kronos_a_share_predictor.backtest.reporting import write_backtest_reports
from kronos_a_share_predictor.clients.kline_client import KlineClient
from kronos_a_share_predictor.clients.recommendation_client import RecommendationClient
from kronos_a_share_predictor.config import AppConfig, load_config
from kronos_a_share_predictor.data.transformers import build_backtest_samples, kline_items_to_frame
from kronos_a_share_predictor.inference.kronos_service import KronosService
from kronos_a_share_predictor.persistence.mysql_repository import MysqlRepository


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kronos backtest and context length scan")
    parser.add_argument("--recommendation-date", help="Override recommendation date, format YYYY-MM-DD")
    parser.add_argument("--backtest-start-date", help="Override backtest start date, format YYYY-MM-DD")
    parser.add_argument("--backtest-end-date", help="Override backtest end date, format YYYY-MM-DD")
    parser.add_argument("--context-lengths", help="Comma-separated context lengths")
    parser.add_argument("--success-mape-threshold", type=float, help="Success threshold for day3 close MAPE")
    parser.add_argument("--report-output-dir", help="Directory to write CSV reports")
    return parser.parse_args()


def _parse_context_lengths(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    updates = {}
    if args.recommendation_date:
        updates["recommendation_date"] = date.fromisoformat(args.recommendation_date)
    if args.backtest_start_date:
        updates["backtest_start_date"] = date.fromisoformat(args.backtest_start_date)
    if args.backtest_end_date:
        updates["backtest_end_date"] = date.fromisoformat(args.backtest_end_date)
    if args.context_lengths:
        updates["backtest_context_lengths"] = _parse_context_lengths(args.context_lengths)
    if args.success_mape_threshold is not None:
        updates["success_mape_threshold"] = args.success_mape_threshold
    if args.report_output_dir:
        updates["report_output_dir"] = config.report_output_dir.__class__(args.report_output_dir)

    next_config = replace(config, **updates) if updates else config
    if args.backtest_end_date and not args.backtest_start_date:
        next_config = replace(next_config, backtest_start_date=next_config.backtest_end_date - timedelta(days=62))
    return next_config


def _fetch_history_by_stock(config: AppConfig, stock_codes: list[str], client: KlineClient) -> tuple[dict[str, object], dict[str, str]]:
    max_context = max(config.backtest_context_lengths)
    fetch_start_date = config.backtest_start_date - timedelta(days=max_context * 2 + config.pred_len * 2)
    history_by_stock = {}
    failures: dict[str, str] = {}

    for stock_code in stock_codes:
        try:
            items = client.fetch_kline(stock_code, fetch_start_date.isoformat(), config.backtest_end_date.isoformat())
            history_by_stock[stock_code] = kline_items_to_frame(stock_code, items)
        except Exception as exc:
            failures[stock_code] = str(exc)
            logger.warning("skip %s because %s", stock_code, exc)

    return history_by_stock, failures


def _print_best_result(summary_frame) -> tuple[int | None, float | None]:
    if summary_frame.empty or int(summary_frame.iloc[0]["sample_count"]) == 0:
        logger.info("no valid backtest samples were produced")
        return None, None

    best_row = summary_frame.iloc[0]
    best_context = int(best_row["context_length"])
    best_success_rate = float(best_row["day3_success_rate"])
    logger.info(
        "best context_length=%s day1_success_rate=%.4f day2_success_rate=%.4f day3_success_rate=%.4f sample_count=%s avg_day1_mape=%.4f avg_day2_mape=%.4f avg_day3_mape=%.4f",
        best_context,
        float(best_row["day1_success_rate"]),
        float(best_row["day2_success_rate"]),
        best_success_rate,
        int(best_row["sample_count"]),
        float(best_row["avg_day1_mape"]),
        float(best_row["avg_day2_mape"]),
        float(best_row["avg_day3_mape"]),
    )
    return best_context, best_success_rate


def run_backtest(config: AppConfig) -> None:
    recommendation_client = RecommendationClient(config.api_base_url, config.request_timeout)
    kline_client = KlineClient(config.api_base_url, config.request_timeout)
    repository = MysqlRepository(config.db_url)
    repository.create_schema()

    run_uuid = str(uuid.uuid4())
    backtest_run_id: int | None = None
    history_failures: dict[str, str] = {}

    try:
        stock_codes, raw_items = recommendation_client.fetch_stock_codes(config.recommendation_date.isoformat())
        logger.info("fetched %s recommendation rows and %s unique stocks for backtest", len(raw_items), len(stock_codes))

        backtest_run_id = repository.create_backtest_run(
            run_uuid=run_uuid,
            recommendation_date=config.recommendation_date,
            backtest_start_date=config.backtest_start_date,
            backtest_end_date=config.backtest_end_date,
            pred_len=config.pred_len,
            candidate_context_lengths=",".join(str(item) for item in config.backtest_context_lengths),
            success_mape_threshold=config.success_mape_threshold,
        )

        history_by_stock, history_failures = _fetch_history_by_stock(config, stock_codes, kline_client)
        samples_by_context = build_backtest_samples(
            history_by_stock=history_by_stock,
            context_lengths=config.backtest_context_lengths,
            backtest_start_date=config.backtest_start_date,
            backtest_end_date=config.backtest_end_date,
            pred_len=config.pred_len,
        )

        service = KronosService(
            repo_path=config.kronos_repo_path,
            tokenizer_id=config.tokenizer_id,
            model_id=config.model_id,
            max_context=config.max_context,
            device=config.device,
        )
        engine = BacktestEngine(
            service=service,
            pred_len=config.pred_len,
            temperature=config.temperature,
            top_p=config.top_p,
            sample_count=config.sample_count,
            success_mape_threshold=config.success_mape_threshold,
            batch_size=config.backtest_batch_size,
            verbose=config.inference_verbose,
        )
        results, summary_frame, detail_frame = engine.run(samples_by_context)
        repository.save_backtest_results(backtest_run_id, detail_frame.to_dict(orient="records"))

        csv_summary_path = None
        csv_detail_path = None
        if config.write_csv_reports:
            csv_summary_path, csv_detail_path = write_backtest_reports(
                summary_frame=summary_frame,
                detail_frame=detail_frame,
                output_dir=config.report_output_dir,
                run_uuid=run_uuid,
            )

        best_context_length, best_success_rate = _print_best_result(summary_frame)
        repository.complete_backtest_run(
            backtest_run_id,
            best_context_length=best_context_length,
            best_success_rate=best_success_rate,
            total_sample_count=sum(len(samples) for samples in samples_by_context.values()),
            total_result_count=len(results),
            csv_summary_path=csv_summary_path,
            csv_detail_path=csv_detail_path,
            status="completed",
            error_message=None if not history_failures else str(history_failures),
        )
    except Exception as exc:
        logger.exception("backtest job failed")
        if backtest_run_id is not None:
            repository.complete_backtest_run(
                backtest_run_id,
                best_context_length=None,
                best_success_rate=None,
                total_sample_count=0,
                total_result_count=0,
                csv_summary_path=None,
                csv_detail_path=None,
                status="failed",
                error_message=str(exc),
            )
        raise


def main() -> None:
    _setup_logging()
    args = _parse_args()
    config = _apply_cli_overrides(load_config(), args)
    run_backtest(config)
