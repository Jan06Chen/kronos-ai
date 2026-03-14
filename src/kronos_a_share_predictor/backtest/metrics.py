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
    day1_mape: float
    day2_mape: float
    day3_mape: float
    close_mae: float
    close_rmse: float
    day1_direction_correct: bool
    day2_direction_correct: bool
    day3_direction_correct: bool
    day1_success: bool
    day2_success: bool
    day3_success: bool
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
    baseline_close = float(sample.baseline_close)

    def _direction_matches(predicted_close: float, observed_close: float) -> bool:
        predicted_delta = predicted_close - baseline_close
        observed_delta = observed_close - baseline_close
        return (
            (predicted_delta == 0 and observed_delta == 0)
            or (predicted_delta > 0 and observed_delta > 0)
            or (predicted_delta < 0 and observed_delta < 0)
        )

    day1_mape = float(abs(float(pred_close.iloc[0]) - float(actual_close.iloc[0])) / max(abs(float(actual_close.iloc[0])), 1e-8))
    day2_mape = float(abs(float(pred_close.iloc[1]) - float(actual_close.iloc[1])) / max(abs(float(actual_close.iloc[1])), 1e-8))
    day3_mape = float(abs(float(pred_close.iloc[2]) - float(actual_close.iloc[2])) / max(abs(float(actual_close.iloc[2])), 1e-8))

    day1_direction_correct = _direction_matches(float(pred_close.iloc[0]), float(actual_close.iloc[0]))
    day2_direction_correct = _direction_matches(float(pred_close.iloc[1]), float(actual_close.iloc[1]))
    day3_direction_correct = _direction_matches(float(pred_close.iloc[2]), float(actual_close.iloc[2]))

    day1_success = bool(day1_direction_correct and day1_mape <= success_mape_threshold)
    day2_success = bool(day2_direction_correct and day2_mape <= success_mape_threshold)
    day3_success = bool(day3_direction_correct and day3_mape <= success_mape_threshold)

    return BacktestEvaluationResult(
        stock_code=sample.stock_code,
        evaluation_date=str(sample.evaluation_date.date()),
        context_length=sample.context_length,
        history_start_date=str(sample.history_start_date.date()),
        history_end_date=str(sample.history_end_date.date()),
        baseline_close=baseline_close,
        prediction_date_1=future_dates[0],
        prediction_date_2=future_dates[1],
        prediction_date_3=future_dates[2],
        pred_close_1=float(pred_close.iloc[0]),
        pred_close_2=float(pred_close.iloc[1]),
        pred_close_3=float(pred_close.iloc[2]),
        actual_close_1=float(actual_close.iloc[0]),
        actual_close_2=float(actual_close.iloc[1]),
        actual_close_3=float(actual_close.iloc[2]),
        day1_mape=day1_mape,
        day2_mape=day2_mape,
        day3_mape=day3_mape,
        close_mae=close_mae,
        close_rmse=close_rmse,
        day1_direction_correct=day1_direction_correct,
        day2_direction_correct=day2_direction_correct,
        day3_direction_correct=day3_direction_correct,
        day1_success=day1_success,
        day2_success=day2_success,
        day3_success=day3_success,
        direction_correct=day3_direction_correct,
        is_success=day3_success,
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
                "day1_success_count": [0] * len(context_lengths),
                "day2_success_count": [0] * len(context_lengths),
                "day3_success_count": [0] * len(context_lengths),
                "success_count": [0] * len(context_lengths),
                "day1_success_rate": [0.0] * len(context_lengths),
                "day2_success_rate": [0.0] * len(context_lengths),
                "day3_success_rate": [0.0] * len(context_lengths),
                "success_rate": [0.0] * len(context_lengths),
                "day1_direction_accuracy": [0.0] * len(context_lengths),
                "day2_direction_accuracy": [0.0] * len(context_lengths),
                "day3_direction_accuracy": [0.0] * len(context_lengths),
                "direction_accuracy": [0.0] * len(context_lengths),
                "avg_day1_mape": [0.0] * len(context_lengths),
                "avg_day2_mape": [0.0] * len(context_lengths),
                "avg_day3_mape": [0.0] * len(context_lengths),
                "avg_close_mae": [0.0] * len(context_lengths),
                "avg_close_rmse": [0.0] * len(context_lengths),
            }
        )

    grouped = detail_frame.groupby("context_length", as_index=False).agg(
        sample_count=("stock_code", "count"),
        day1_success_count=("day1_success", "sum"),
        day2_success_count=("day2_success", "sum"),
        day3_success_count=("day3_success", "sum"),
        day1_success_rate=("day1_success", "mean"),
        day2_success_rate=("day2_success", "mean"),
        day3_success_rate=("day3_success", "mean"),
        day1_direction_accuracy=("day1_direction_correct", "mean"),
        day2_direction_accuracy=("day2_direction_correct", "mean"),
        day3_direction_accuracy=("day3_direction_correct", "mean"),
        avg_day1_mape=("day1_mape", "mean"),
        avg_day2_mape=("day2_mape", "mean"),
        avg_day3_mape=("day3_mape", "mean"),
        avg_close_mae=("close_mae", "mean"),
        avg_close_rmse=("close_rmse", "mean"),
    )
    grouped["success_count"] = grouped["day3_success_count"]
    grouped["success_rate"] = grouped["day3_success_rate"]
    grouped["direction_accuracy"] = grouped["day3_direction_accuracy"]
    return grouped.sort_values(by=["day3_success_rate", "avg_day3_mape", "sample_count"], ascending=[False, True, False]).reset_index(drop=True)
