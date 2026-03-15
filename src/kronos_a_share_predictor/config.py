from __future__ import annotations

import os
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _shift_months(anchor: date, months: int) -> date:
    year = anchor.year
    month = anchor.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(anchor.day, monthrange(year, month)[1])
    return date(year, month, day)


def _parse_int_list(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        return default
    return tuple(int(item) for item in raw)


def _parse_float_list(value: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        return default
    return tuple(float(item) for item in raw)


def _parse_bool(value: str, default: bool) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppConfig:
    api_base_url: str
    recommendation_date: date
    starttime: date
    endtime: date
    lookback_days: int
    pred_len: int
    min_history_points: int
    tokenizer_id: str
    model_id: str
    kronos_repo_path: Path
    db_url: str
    request_timeout: int
    device: str | None
    top_p: float
    temperature: float
    sample_count: int
    inference_verbose: bool
    backtest_start_date: date
    backtest_end_date: date
    backtest_months: int
    backtest_context_lengths: tuple[int, ...]
    tuning_context_length: int
    tuning_temperatures: tuple[float, ...]
    tuning_top_ps: tuple[float, ...]
    tuning_sample_counts: tuple[int, ...]
    success_mape_threshold: float
    backtest_batch_size: int
    report_output_dir: Path
    write_csv_reports: bool
    max_context: int = 512


def load_config() -> AppConfig:
    load_dotenv()

    endtime = _parse_date(os.getenv("KRONOS_ENDTIME", "2026-03-14"))
    lookback_days = int(os.getenv("KRONOS_LOOKBACK_DAYS", "200"))
    starttime_raw = os.getenv("KRONOS_STARTTIME", "").strip()
    starttime = _parse_date(starttime_raw) if starttime_raw else endtime - timedelta(days=lookback_days)
    backtest_months = int(os.getenv("KRONOS_BACKTEST_MONTHS", "2"))
    backtest_end_date = _parse_date(os.getenv("KRONOS_BACKTEST_END_DATE", endtime.isoformat()))
    backtest_start_raw = os.getenv("KRONOS_BACKTEST_START_DATE", "").strip()
    backtest_start_date = _parse_date(backtest_start_raw) if backtest_start_raw else _shift_months(backtest_end_date, backtest_months)

    device_raw = os.getenv("KRONOS_DEVICE", "").strip()
    return AppConfig(
        api_base_url=os.getenv("KRONOS_API_BASE_URL", "http://localhost:5000/api/v1").rstrip("/"),
        recommendation_date=_parse_date(os.getenv("KRONOS_RECOMMENDATION_DATE", "2026-03-13")),
        starttime=starttime,
        endtime=endtime,
        lookback_days=lookback_days,
        pred_len=int(os.getenv("KRONOS_PRED_LEN", "3")),
        min_history_points=int(os.getenv("KRONOS_MIN_HISTORY_POINTS", "60")),
        tokenizer_id=os.getenv("KRONOS_TOKENIZER_ID", "NeoQuasar/Kronos-Tokenizer-base"),
        model_id=os.getenv("KRONOS_MODEL_ID", "NeoQuasar/Kronos-base"),
        kronos_repo_path=Path(os.getenv("KRONOS_REPO_PATH", "vendor/Kronos")).expanduser(),
        db_url=os.getenv(
            "KRONOS_DB_URL",
            "mysql+pymysql://root:9801309@127.0.0.1:3306/stock_data?charset=utf8mb4",
        ),
        request_timeout=int(os.getenv("KRONOS_REQUEST_TIMEOUT", "30")),
        device=device_raw or None,
        top_p=float(os.getenv("KRONOS_TOP_P", "0.9")),
        temperature=float(os.getenv("KRONOS_TEMPERATURE", "1.0")),
        sample_count=int(os.getenv("KRONOS_SAMPLE_COUNT", "1")),
        inference_verbose=_parse_bool(os.getenv("KRONOS_INFERENCE_VERBOSE", "false"), False),
        backtest_start_date=backtest_start_date,
        backtest_end_date=backtest_end_date,
        backtest_months=backtest_months,
        backtest_context_lengths=_parse_int_list(
            os.getenv("KRONOS_BACKTEST_CONTEXT_LENGTHS", "30,60,90,120,150,180,200,300,400,500"),
            (30, 60, 90, 120, 150, 180, 200, 300, 400, 500),
        ),
        tuning_context_length=int(os.getenv("KRONOS_TUNING_CONTEXT_LENGTH", "120")),
        tuning_temperatures=_parse_float_list(
            os.getenv("KRONOS_TUNING_TEMPERATURES", "0.7,0.9,1.0,1.1"),
            (0.7, 0.9, 1.0, 1.1),
        ),
        tuning_top_ps=_parse_float_list(
            os.getenv("KRONOS_TUNING_TOP_PS", "0.8,0.9,0.95"),
            (0.8, 0.9, 0.95),
        ),
        tuning_sample_counts=_parse_int_list(
            os.getenv("KRONOS_TUNING_SAMPLE_COUNTS", "1,3,5"),
            (1, 3, 5),
        ),
        success_mape_threshold=float(os.getenv("KRONOS_BACKTEST_SUCCESS_MAPE_THRESHOLD", "0.08")),
        backtest_batch_size=int(os.getenv("KRONOS_BACKTEST_BATCH_SIZE", "32")),
        report_output_dir=Path(os.getenv("KRONOS_REPORT_OUTPUT_DIR", "outputs/reports")).expanduser(),
        write_csv_reports=_parse_bool(os.getenv("KRONOS_BACKTEST_WRITE_CSV", "true"), True),
    )
