from __future__ import annotations

import argparse
import logging
import uuid
from dataclasses import replace
from datetime import date, timedelta

from kronos_a_share_predictor.clients.recommendation_client import RecommendationClient
from kronos_a_share_predictor.config import AppConfig, load_config
from kronos_a_share_predictor.data.transformers import prepare_series_batch
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
    parser.add_argument("--lookback-days", type=int, help="Override historical lookback days")
    parser.add_argument("--pred-len", type=int, help="Override prediction horizon length")
    parser.add_argument("--min-history-points", type=int, help="Override minimum history points required")
    parser.add_argument("--temperature", type=float, help="Override prediction temperature")
    parser.add_argument("--top-p", type=float, help="Override prediction top_p")
    parser.add_argument("--sample-count", type=int, help="Override prediction sample count")
    parser.add_argument("--device", help="Override inference device, e.g. cpu, cuda, mps")
    parser.add_argument("--verbose-inference", action="store_true", help="Show Kronos autoregressive progress")
    args, unknown = parser.parse_known_args()

    invalid_options = [item for item in unknown if item.startswith("-") and item != "-"]
    if invalid_options:
        parser.error(f"unrecognized arguments: {' '.join(invalid_options)}")
    if unknown:
        logger.warning("ignore unexpected trailing arguments: %s", " ".join(unknown))

    return args


def _apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    updates = {}
    if args.recommendation_date:
        updates["recommendation_date"] = date.fromisoformat(args.recommendation_date)
        updates["use_recommendation_source"] = True
    if args.starttime:
        updates["starttime"] = date.fromisoformat(args.starttime)
    if args.endtime:
        updates["endtime"] = date.fromisoformat(args.endtime)
    if args.context_length is not None:
        if args.context_length <= 0:
            raise ValueError("context-length 必须是正整数")
        updates["prediction_context_length"] = args.context_length
    if args.lookback_days is not None:
        if args.lookback_days <= 0:
            raise ValueError("lookback-days 必须是正整数")
        updates["lookback_days"] = args.lookback_days
    if args.pred_len is not None:
        if args.pred_len <= 0:
            raise ValueError("pred-len 必须是正整数")
        updates["pred_len"] = args.pred_len
    if args.min_history_points is not None:
        if args.min_history_points <= 0:
            raise ValueError("min-history-points 必须是正整数")
        updates["min_history_points"] = args.min_history_points
    if args.temperature is not None:
        if args.temperature <= 0:
            raise ValueError("temperature 必须大于 0")
        updates["temperature"] = args.temperature
    if args.top_p is not None:
        if not 0 < args.top_p <= 1:
            raise ValueError("top-p 必须在 (0, 1] 区间内")
        updates["top_p"] = args.top_p
    if args.sample_count is not None:
        if args.sample_count <= 0:
            raise ValueError("sample-count 必须是正整数")
        updates["sample_count"] = args.sample_count
    if args.device:
        updates["device"] = args.device
    if args.verbose_inference:
        updates["inference_verbose"] = True

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


def _resolve_fetch_start_date(config: AppConfig, effective_context_length: int) -> date:
    required_points = max(effective_context_length, config.min_history_points) + config.pred_len
    buffered_start_date = config.endtime - timedelta(days=required_points * 2)
    return min(config.starttime, buffered_start_date)


def run_job(config: AppConfig) -> None:
    recommendation_client = RecommendationClient(config.api_base_url, config.request_timeout)
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
    effective_recommendation_date = config.recommendation_date if config.use_recommendation_source else config.endtime
    fetch_start_date = _resolve_fetch_start_date(config, effective_context_length)

    try:
        stock_codes, raw_items = recommendation_client.fetch_stock_codes(
            config.recommendation_date.isoformat() if config.use_recommendation_source else None
        )
        logger.info(
            "fetched %s source rows and %s unique stocks from %s",
            len(raw_items),
            len(stock_codes),
            "recommendations" if config.use_recommendation_source else "stock/list",
        )

        run_id = repository.create_run(
            run_uuid=run_uuid,
            recommendation_date=effective_recommendation_date,
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

        history_by_stock, history_failures = repository.fetch_history_by_stock(
            stock_codes,
            fetch_start_date.isoformat(),
            config.endtime.isoformat(),
        )
        for stock_code, error_message in history_failures.items():
            stock_failures[stock_code] = error_message
            logger.warning(
                "skip %s because %s, requested_range=%s..%s, fetch_range=%s..%s, source=database:all_detail_day",
                stock_code,
                error_message,
                config.starttime.isoformat(),
                config.endtime.isoformat(),
                fetch_start_date.isoformat(),
                config.endtime.isoformat(),
            )

        eligible_history_count = sum(1 for frame in history_by_stock.values() if len(frame) >= config.min_history_points)
        logger.info(
            "history summary requested_range=%s..%s fetch_range=%s..%s loaded_stocks=%s eligible_stocks=%s history_failures=%s min_history_points=%s context_length=%s",
            config.starttime.isoformat(),
            config.endtime.isoformat(),
            fetch_start_date.isoformat(),
            config.endtime.isoformat(),
            len(history_by_stock),
            eligible_history_count,
            len(history_failures),
            config.min_history_points,
            effective_context_length,
        )

        max_available_history = max((len(frame) for frame in history_by_stock.values()), default=0)
        if max_available_history < config.min_history_points:
            logger.warning(
                "no eligible stocks for prediction because max_available_history=%s is below min_history_points=%s, requested_range=%s..%s, fetch_range=%s..%s",
                max_available_history,
                config.min_history_points,
                config.starttime.isoformat(),
                config.endtime.isoformat(),
                fetch_start_date.isoformat(),
                config.endtime.isoformat(),
            )
            repository.complete_run(
                run_id=run_id,
                success_count=0,
                failed_count=len(stock_codes),
                status="completed",
                error_message={
                    "message": "没有任何股票满足最小历史数据要求",
                    "requested_range": f"{config.starttime.isoformat()}..{config.endtime.isoformat()}",
                    "fetch_range": f"{fetch_start_date.isoformat()}..{config.endtime.isoformat()}",
                    "min_history_points": config.min_history_points,
                    "max_available_history": max_available_history,
                    "eligible_stock_count": eligible_history_count,
                    "history_fetch_failure_count": len(history_failures),
                },
            )
            return

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
                # logger.info("predicted and persisted stock=%s count=1", stock_code)
            except Exception as exc:
                stock_failures[stock_code] = str(exc)
                logger.warning("predict failed for %s because %s", stock_code, exc)

        if success_count == 0:
            logger.warning(
                "run completed without predictions, success=%s failed=%s requested_range=%s..%s fetch_range=%s..%s max_available_history=%s min_history_points=%s",
                success_count,
                len(stock_failures),
                config.starttime.isoformat(),
                config.endtime.isoformat(),
                fetch_start_date.isoformat(),
                config.endtime.isoformat(),
                max_available_history,
                config.min_history_points,
            )
            repository.complete_run(
                run_id=run_id,
                success_count=0,
                failed_count=len(stock_failures),
                status="completed",
                error_message={
                    "message": "没有任何股票满足预测条件或预测成功",
                    "requested_range": f"{config.starttime.isoformat()}..{config.endtime.isoformat()}",
                    "fetch_range": f"{fetch_start_date.isoformat()}..{config.endtime.isoformat()}",
                    "min_history_points": config.min_history_points,
                    "max_available_history": max_available_history,
                    "failure_count": len(stock_failures),
                },
            )
            return

        repository.complete_run(
            run_id=run_id,
            success_count=success_count,
            failed_count=len(stock_failures),
            status="completed",
            error_message=None
            if not stock_failures
            else {
                "message": "部分股票预测失败",
                "failure_count": len(stock_failures),
                "sample_failures": stock_failures,
            },
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
