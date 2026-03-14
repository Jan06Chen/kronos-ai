from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import pandas as pd


@dataclass(frozen=True)
class BacktestEvaluationResult:
    stock_code: str
    evaluation_date: str
    context_length: int
    history_start_date: str
    history_end_date: str
    baseline_close: float
    prediction_date_1: str
    prediction_date_2: str
    prediction_date_3: str
    pred_close_1: float
    pred_close_2: float
    pred_close_3: float
    actual_close_1: float
    actual_close_2: float
    actual_close_3: float
    day3_mape: float
    close_mae: float
    close_rmse: float
    direction_correct: bool
    is_success: bool

    def to_record(self) -> dict:
        return asdict(self)


def evaluate_backtest_prediction(sample, prediction_frame: pd.DataFrame, success_mape_threshold: float) -> BacktestEvaluationResult:
    pred_close = prediction_frame["close"].reset_index(drop=True).astype(float)
    actual_close = sample.actual_future_df["close"].reset_index(drop=True).astype(float)
    future_dates = pd.to_datetime(sample.actual_future_df["timestamps"]).dt.date.astype(str).tolist()
    abs_errors = (pred_close - actual_close).abs()
    close_mae = float(abs_errors.mean())
    close_rmse = float(sqrt(((pred_close - actual_close) ** 2).mean()))
    actual_day3 = float(actual_close.iloc[2])
    pred_day3 = float(pred_close.iloc[2])
    day3_mape = float(abs(pred_day3 - actual_day3) / max(abs(actual_day3), 1e-8))

    pred_delta = pred_day3 - float(sample.baseline_close)
    actual_delta = actual_day3 - float(sample.baseline_close)
    direction_correct = (pred_delta == 0 and actual_delta == 0) or (pred_delta > 0 and actual_delta > 0) or (pred_delta < 0 and actual_delta < 0)
    is_success = bool(direction_correct and day3_mape <= success_mape_threshold)

    return BacktestEvaluationResult(
        stock_code=sample.stock_code,
        evaluation_date=str(sample.evaluation_date.date()),
        context_length=sample.context_length,
        history_start_date=str(sample.history_start_date.date()),
        history_end_date=str(sample.history_end_date.date()),
        baseline_close=float(sample.baseline_close),
        prediction_date_1=future_dates[0],
        prediction_date_2=future_dates[1],
        prediction_date_3=future_dates[2],
        pred_close_1=float(pred_close.iloc[0]),
        pred_close_2=float(pred_close.iloc[1]),
        pred_close_3=float(pred_close.iloc[2]),
        actual_close_1=float(actual_close.iloc[0]),
        actual_close_2=float(actual_close.iloc[1]),
        actual_close_3=float(actual_close.iloc[2]),
        day3_mape=day3_mape,
        close_mae=close_mae,
        close_rmse=close_rmse,
        direction_correct=direction_correct,
        is_success=is_success,
    )


def results_to_detail_frame(results: list[BacktestEvaluationResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([result.to_record() for result in results]).sort_values(
        by=["context_length", "evaluation_date", "stock_code"]
    ).reset_index(drop=True)


def summarize_results(results: list[BacktestEvaluationResult], context_lengths: tuple[int, ...]) -> pd.DataFrame:
    detail_frame = results_to_detail_frame(results)
    if detail_frame.empty:
        return pd.DataFrame(
            {
                "context_length": list(context_lengths),
                "sample_count": [0] * len(context_lengths),
                "success_count": [0] * len(context_lengths),
                "success_rate": [0.0] * len(context_lengths),
                "direction_accuracy": [0.0] * len(context_lengths),
                "avg_day3_mape": [0.0] * len(context_lengths),
                "avg_close_mae": [0.0] * len(context_lengths),
                "avg_close_rmse": [0.0] * len(context_lengths),
            }
        )

    grouped = detail_frame.groupby("context_length", as_index=False).agg(
        sample_count=("stock_code", "count"),
        success_count=("is_success", "sum"),
        success_rate=("is_success", "mean"),
        direction_accuracy=("direction_correct", "mean"),
        avg_day3_mape=("day3_mape", "mean"),
        avg_close_mae=("close_mae", "mean"),
        avg_close_rmse=("close_rmse", "mean"),
    )
    return grouped.sort_values(by=["success_rate", "avg_day3_mape", "sample_count"], ascending=[False, True, False]).reset_index(drop=True)
