from __future__ import annotations

import argparse
import logging
import uuid
from dataclasses import replace
from datetime import date, timedelta
from itertools import product

import pandas as pd

from kronos_a_share_predictor.backtest.engine import BacktestEngine
from kronos_a_share_predictor.backtest.reporting import (
    write_sampling_tuning_reports,
    append_sampling_summary_row,
    append_sampling_details_frame,
    write_best_detail_file,
)
from kronos_a_share_predictor.clients.kline_client import KlineClient
from kronos_a_share_predictor.clients.recommendation_client import RecommendationClient
from kronos_a_share_predictor.config import AppConfig, load_config
from kronos_a_share_predictor.data.transformers import build_backtest_samples, kline_items_to_frame
from kronos_a_share_predictor.inference.kronos_service import KronosService


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Kronos sampling parameters for fixed context length")
    parser.add_argument("--recommendation-date", help="Override recommendation date, format YYYY-MM-DD")
    parser.add_argument("--backtest-start-date", help="Override backtest start date, format YYYY-MM-DD")
    parser.add_argument("--backtest-end-date", help="Override backtest end date, format YYYY-MM-DD")
    parser.add_argument("--context-length", type=int, help="Fixed context length for sampling-parameter tuning")
    parser.add_argument("--temperatures", help="Comma-separated temperature candidates")
    parser.add_argument("--top-ps", help="Comma-separated top_p candidates")
    parser.add_argument("--sample-counts", help="Comma-separated sample_count candidates")
    parser.add_argument("--success-mape-threshold", type=float, help="Success threshold for day3 close MAPE")
    parser.add_argument("--report-output-dir", help="Directory to write CSV reports")
    parser.add_argument("--verbose-inference", action="store_true", help="Show Kronos autoregressive progress")
    return parser.parse_args()


def _apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    updates = {}
    if args.recommendation_date:
        updates["recommendation_date"] = date.fromisoformat(args.recommendation_date)
    if args.backtest_start_date:
        updates["backtest_start_date"] = date.fromisoformat(args.backtest_start_date)
    if args.backtest_end_date:
        updates["backtest_end_date"] = date.fromisoformat(args.backtest_end_date)
    if args.context_length is not None:
        updates["tuning_context_length"] = args.context_length
    if args.temperatures:
        updates["tuning_temperatures"] = _parse_float_list(args.temperatures)
    if args.top_ps:
        updates["tuning_top_ps"] = _parse_float_list(args.top_ps)
    if args.sample_counts:
        updates["tuning_sample_counts"] = _parse_int_list(args.sample_counts)
    if args.success_mape_threshold is not None:
        updates["success_mape_threshold"] = args.success_mape_threshold
    if args.report_output_dir:
        updates["report_output_dir"] = config.report_output_dir.__class__(args.report_output_dir)
    if args.verbose_inference:
        updates["inference_verbose"] = True

    next_config = replace(config, **updates) if updates else config
    if args.backtest_end_date and not args.backtest_start_date:
        next_config = replace(next_config, backtest_start_date=next_config.backtest_end_date - timedelta(days=62))
    return next_config


def _fetch_history_by_stock(config: AppConfig, stock_codes: list[str], client: KlineClient) -> tuple[dict[str, object], dict[str, str]]:
    fetch_start_date = config.backtest_start_date - timedelta(days=config.tuning_context_length * 2 + config.pred_len * 2)
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


def _build_summary_record(summary_row: pd.Series, *, temperature: float, top_p: float, sample_count: int, verbose: bool) -> dict:
    record = summary_row.to_dict()
    record["temperature"] = temperature
    record["top_p"] = top_p
    record["sample_count_candidate"] = sample_count
    record["verbose"] = verbose
    return record


def _print_best_configuration(summary_frame: pd.DataFrame) -> None:
    if summary_frame.empty:
        logger.info("no valid tuning result rows were produced")
        return

    best_row = summary_frame.iloc[0]
    logger.info(
        "best sampling params context_length=%s temperature=%.3f top_p=%.3f sample_count=%s day1_success_rate=%.4f day2_success_rate=%.4f day3_success_rate=%.4f avg_day3_mape=%.4f",
        int(best_row["context_length"]),
        float(best_row["temperature"]),
        float(best_row["top_p"]),
        int(best_row["sample_count_candidate"]),
        float(best_row["day1_success_rate"]),
        float(best_row["day2_success_rate"]),
        float(best_row["day3_success_rate"]),
        float(best_row["avg_day3_mape"]),
    )


def run_tuning(config: AppConfig) -> None:
    recommendation_client = RecommendationClient(config.api_base_url, config.request_timeout)
    kline_client = KlineClient(config.api_base_url, config.request_timeout)

    stock_codes, raw_items = recommendation_client.fetch_stock_codes(config.recommendation_date.isoformat())
    logger.info("fetched %s recommendation rows and %s unique stocks for sampling tuning", len(raw_items), len(stock_codes))

    history_by_stock, history_failures = _fetch_history_by_stock(config, stock_codes, kline_client)
    if history_failures:
        logger.info("history fetch failures=%s", len(history_failures))

    samples_by_context = build_backtest_samples(
        history_by_stock=history_by_stock,
        context_lengths=(config.tuning_context_length,),
        backtest_start_date=config.backtest_start_date,
        backtest_end_date=config.backtest_end_date,
        pred_len=config.pred_len,
    )
    fixed_samples = samples_by_context.get(config.tuning_context_length, [])
    if not fixed_samples:
        raise RuntimeError(f"固定 context_length={config.tuning_context_length} 没有生成任何有效回测样本")

    service = KronosService(
        repo_path=config.kronos_repo_path,
        tokenizer_id=config.tokenizer_id,
        model_id=config.model_id,
        max_context=config.max_context,
        device=config.device,
    )

    combo_records: list[dict] = []
    best_detail_frame = pd.DataFrame()
    best_sort_key: tuple[float, float, int] | None = None
    run_uuid = str(uuid.uuid4())

    for temperature, top_p, sample_count in product(
        config.tuning_temperatures,
        config.tuning_top_ps,
        config.tuning_sample_counts,
    ):
        logger.info(
            "tuning context_length=%s temperature=%.3f top_p=%.3f sample_count=%s verbose=%s",
            config.tuning_context_length,
            temperature,
            top_p,
            sample_count,
            config.inference_verbose,
        )
        engine = BacktestEngine(
            service=service,
            pred_len=config.pred_len,
            temperature=temperature,
            top_p=top_p,
            sample_count=sample_count,
            success_mape_threshold=config.success_mape_threshold,
            verbose=config.inference_verbose,
        )
        _, summary_frame, detail_frame = engine.run({config.tuning_context_length: fixed_samples})
        if summary_frame.empty:
            continue
        summary_row = summary_frame.iloc[0]
        record = _build_summary_record(
            summary_row,
            temperature=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=config.inference_verbose,
        )
        combo_records.append(record)
        try:
            append_sampling_summary_row(record, config.report_output_dir, run_uuid)
        except Exception:
            logger.exception("failed to append sampling summary row to disk")
        sort_key = (
            float(summary_row["day3_success_rate"]),
            -float(summary_row["avg_day3_mape"]),
            int(summary_row["sample_count"]),
        )
        # append per-combination detail rows to disk
        try:
            detail_to_write = detail_frame.assign(
                temperature=temperature,
                top_p=top_p,
                sample_count_candidate=sample_count,
                verbose=config.inference_verbose,
            )
            append_sampling_details_frame(detail_to_write, config.report_output_dir, run_uuid)
        except Exception:
            logger.exception("failed to append sampling detail frame to disk")

        if best_sort_key is None or sort_key > best_sort_key:
            best_sort_key = sort_key
            best_detail_frame = detail_to_write
            try:
                write_best_detail_file(best_detail_frame, config.report_output_dir, run_uuid)
            except Exception:
                logger.exception("failed to write best detail file to disk")

    if not combo_records:
        raise RuntimeError("没有产生任何有效的采样参数调优结果")

    summary_output = pd.DataFrame(combo_records).sort_values(
        by=["day3_success_rate", "avg_day3_mape", "sample_count"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    if config.write_csv_reports:
        summary_path, detail_path = write_sampling_tuning_reports(
            summary_frame=summary_output,
            best_detail_frame=best_detail_frame,
            output_dir=config.report_output_dir,
            run_uuid=run_uuid,
        )
        logger.info("sampling tuning reports written summary=%s best_details=%s", summary_path, detail_path)

    _print_best_configuration(summary_output)


def main() -> None:
    _setup_logging()
    args = _parse_args()
    config = _apply_cli_overrides(load_config(), args)
    run_tuning(config)
