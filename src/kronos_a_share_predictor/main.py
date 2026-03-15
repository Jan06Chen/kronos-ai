from __future__ import annotations

import argparse
import logging
import uuid
from dataclasses import replace
from datetime import date, timedelta

from kronos_a_share_predictor.clients.kline_client import KlineClient
from kronos_a_share_predictor.clients.recommendation_client import RecommendationClient
from kronos_a_share_predictor.config import AppConfig, load_config
from kronos_a_share_predictor.data.transformers import kline_items_to_frame, prepare_series_batch
from kronos_a_share_predictor.inference.kronos_service import KronosService
from kronos_a_share_predictor.persistence.mysql_repository import MysqlRepository


logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Kronos A-share prediction job")
    parser.add_argument("--recommendation-date", help="Override recommendation date, format YYYY-MM-DD")
    parser.add_argument("--starttime", help="Override starttime, format YYYY-MM-DD")
    parser.add_argument("--endtime", help="Override endtime, format YYYY-MM-DD")
    parser.add_argument("--context-length", type=int, help="Override prediction context length")
    return parser.parse_args()


def _apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    updates = {}
    if args.recommendation_date:
        updates["recommendation_date"] = date.fromisoformat(args.recommendation_date)
    if args.starttime:
        updates["starttime"] = date.fromisoformat(args.starttime)
    if args.endtime:
        updates["endtime"] = date.fromisoformat(args.endtime)
    if args.context_length is not None:
        if args.context_length <= 0:
            raise ValueError("context-length 必须是正整数")
        updates["prediction_context_length"] = args.context_length

    next_config = replace(config, **updates) if updates else config
    if args.endtime and not args.starttime:
        next_config = replace(next_config, starttime=next_config.endtime - timedelta(days=next_config.lookback_days))
    return next_config


def _build_prediction_rows(predictions) -> list[dict]:
    rows: list[dict] = []
    for prepared, frame in predictions:
        indexed = frame.reset_index()
        timestamp_column = indexed.columns[0]
        indexed = indexed.rename(columns={timestamp_column: "prediction_date"})
        for record in indexed.to_dict(orient="records"):
            rows.append(
                {
                    "stock_code": prepared.stock_code,
                    "history_start_date": prepared.history_start_date.date(),
                    "history_end_date": prepared.history_end_date.date(),
                    "prediction_date": record["prediction_date"].date(),
                    "pred_open": float(record["open"]),
                    "pred_high": float(record["high"]),
                    "pred_low": float(record["low"]),
                    "pred_close": float(record["close"]),
                    "pred_volume": float(record["volume"]),
                    "pred_amount": float(record["amount"]),
                }
            )
    return rows


def run_job(config: AppConfig) -> None:
    recommendation_client = RecommendationClient(config.api_base_url, config.request_timeout)
    kline_client = KlineClient(config.api_base_url, config.request_timeout)
    repository = MysqlRepository(config.db_url)
    repository.create_schema()

    run_uuid = str(uuid.uuid4())
    run_id: int | None = None
    stock_failures: dict[str, str] = {}
    success_count = 0
    effective_context_length = min(
        config.lookback_days,
        config.max_context,
        config.prediction_context_length if config.prediction_context_length is not None else config.max_context,
    )

    try:
        stock_codes, raw_items = recommendation_client.fetch_stock_codes(config.recommendation_date.isoformat())
        logger.info("fetched %s recommendation rows and %s unique stocks", len(raw_items), len(stock_codes))

        run_id = repository.create_run(
            run_uuid=run_uuid,
            recommendation_date=config.recommendation_date,
            starttime=config.starttime,
            endtime=config.endtime,
            lookback_days=config.lookback_days,
            pred_len=config.pred_len,
            min_history_points=config.min_history_points,
            max_context=config.max_context,
            context_length=effective_context_length,
            model_name=config.model_id,
            tokenizer_name=config.tokenizer_id,
            device=config.device,
            temperature=config.temperature,
            top_p=config.top_p,
            sample_count=config.sample_count,
            inference_verbose=config.inference_verbose,
            stock_count_raw=len(raw_items),
            stock_count_dedup=len(stock_codes),
        )

        history_by_stock = {}
        for stock_code in stock_codes:
            try:
                items = kline_client.fetch_kline(stock_code, config.starttime.isoformat(), config.endtime.isoformat())
                history_by_stock[stock_code] = kline_items_to_frame(stock_code, items)
            except Exception as exc:
                stock_failures[stock_code] = str(exc)
                logger.warning("skip %s because %s", stock_code, exc)

        service = KronosService(
            repo_path=config.kronos_repo_path,
            tokenizer_id=config.tokenizer_id,
            model_id=config.model_id,
            max_context=config.max_context,
            device=config.device,
        )

        for stock_code, frame in history_by_stock.items():
            prepared_series, preparation_failures = prepare_series_batch(
                history_by_stock={stock_code: frame},
                pred_len=config.pred_len,
                max_context=effective_context_length,
                min_history_points=config.min_history_points,
            )
            if preparation_failures:
                stock_failures.update(preparation_failures)
                continue
            if not prepared_series:
                stock_failures[stock_code] = "单股票预处理后没有可用的推理序列"
                continue

            try:
                prediction = service.predict(
                    prepared_series=prepared_series[0],
                    pred_len=config.pred_len,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    sample_count=config.sample_count,
                    verbose=config.inference_verbose,
                )
                repository.save_predictions(run_id, _build_prediction_rows([prediction]))
                success_count += 1
                logger.info("predicted and persisted stock=%s count=1", stock_code)
            except Exception as exc:
                stock_failures[stock_code] = str(exc)
                logger.warning("predict failed for %s because %s", stock_code, exc)

        if success_count == 0:
            raise RuntimeError("没有任何股票预测成功，无法完成本次 Kronos 预测")

        repository.complete_run(
            run_id=run_id,
            success_count=success_count,
            failed_count=len(stock_failures),
            status="completed",
            error_message=None if not stock_failures else str(stock_failures),
        )
        logger.info(
            "run completed, success=%s failed=%s run_uuid=%s",
            success_count,
            len(stock_failures),
            run_uuid,
        )
    except Exception as exc:
        logger.exception("prediction job failed")
        if run_id is not None:
            repository.complete_run(
                run_id=run_id,
                success_count=0,
                failed_count=len(stock_failures),
                status="failed",
                error_message=str(exc),
            )
        raise


def main() -> None:
    _setup_logging()
    args = _parse_args()
    config = _apply_cli_overrides(load_config(), args)
    run_job(config)
