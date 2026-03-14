from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


REQUIRED_COLUMNS = ["open", "high", "low", "close"]
OPTIONAL_COLUMNS = ["volume", "amount"]


@dataclass(frozen=True)
class PreparedSeries:
    stock_code: str
    history_start_date: pd.Timestamp
    history_end_date: pd.Timestamp
    x_df: pd.DataFrame
    x_timestamp: pd.Series
    y_timestamp: pd.Series


@dataclass(frozen=True)
class BacktestSample:
    stock_code: str
    evaluation_date: pd.Timestamp
    context_length: int
    history_start_date: pd.Timestamp
    history_end_date: pd.Timestamp
    baseline_close: float
    x_df: pd.DataFrame
    x_timestamp: pd.Series
    y_timestamp: pd.Series
    actual_future_df: pd.DataFrame


def kline_items_to_frame(stock_code: str, items: list[dict]) -> pd.DataFrame:
    if not items:
        raise ValueError(f"股票 {stock_code} 没有返回任何 K 线数据")

    frame = pd.DataFrame(items).copy()
    if "date" not in frame.columns:
        raise ValueError(f"股票 {stock_code} 的 K 线数据缺少 date 字段")

    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(f"股票 {stock_code} 的 OHLC 字段存在空值")

    frame["volume"] = frame["volume"].fillna(0.0)
    frame["amount"] = frame["amount"].fillna(0.0)
    frame["stock_code"] = stock_code
    frame["timestamps"] = frame["date"]
    return frame[["stock_code", "date", "timestamps", *REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]].copy()


def build_future_timestamps(last_timestamp: pd.Timestamp, pred_len: int) -> pd.Series:
    future_range = pd.bdate_range(start=last_timestamp + pd.offsets.BDay(1), periods=pred_len)
    return pd.Series(future_range, name="timestamps")


def prepare_series_batch(
    history_by_stock: dict[str, pd.DataFrame],
    pred_len: int,
    max_context: int,
    min_history_points: int,
) -> tuple[list[PreparedSeries], dict[str, str]]:
    failures: dict[str, str] = {}
    valid_histories: dict[str, pd.DataFrame] = {}

    for stock_code, frame in history_by_stock.items():
        if len(frame) < min_history_points:
            failures[stock_code] = f"历史数据不足，至少需要 {min_history_points} 条，实际 {len(frame)} 条"
            continue
        valid_histories[stock_code] = frame

    if not valid_histories:
        return [], failures

    common_length = min(len(frame) for frame in valid_histories.values())
    common_length = min(common_length, max_context)

    if common_length < min_history_points:
        for stock_code in valid_histories:
            failures[stock_code] = f"对齐后共同长度仅 {common_length} 条，低于最小阈值 {min_history_points}"
        return [], failures

    prepared_list: list[PreparedSeries] = []
    for stock_code, frame in valid_histories.items():
        clipped = frame.tail(common_length).reset_index(drop=True)
        history_end = clipped["timestamps"].iloc[-1]
        prepared_list.append(
            PreparedSeries(
                stock_code=stock_code,
                history_start_date=clipped["timestamps"].iloc[0],
                history_end_date=history_end,
                x_df=clipped[[*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]].reset_index(drop=True),
                x_timestamp=clipped["timestamps"].reset_index(drop=True),
                y_timestamp=build_future_timestamps(history_end, pred_len),
            )
        )

    return prepared_list, failures


def build_backtest_samples(
    history_by_stock: dict[str, pd.DataFrame],
    context_lengths: tuple[int, ...],
    backtest_start_date,
    backtest_end_date,
    pred_len: int,
) -> dict[int, list[BacktestSample]]:
    samples_by_context = {context_length: [] for context_length in context_lengths}

    for stock_code, frame in history_by_stock.items():
        ordered = frame.sort_values("timestamps").reset_index(drop=True)
        if len(ordered) <= pred_len:
            continue

        for evaluation_index in range(len(ordered) - pred_len):
            evaluation_row = ordered.iloc[evaluation_index]
            evaluation_date = evaluation_row["timestamps"]
            if evaluation_date.date() < backtest_start_date or evaluation_date.date() > backtest_end_date:
                continue

            actual_future = ordered.iloc[evaluation_index + 1 : evaluation_index + 1 + pred_len].reset_index(drop=True)
            if len(actual_future) < pred_len:
                continue

            for context_length in context_lengths:
                if evaluation_index + 1 < context_length:
                    continue

                history_window = ordered.iloc[evaluation_index + 1 - context_length : evaluation_index + 1].reset_index(drop=True)
                samples_by_context[context_length].append(
                    BacktestSample(
                        stock_code=stock_code,
                        evaluation_date=evaluation_date,
                        context_length=context_length,
                        history_start_date=history_window["timestamps"].iloc[0],
                        history_end_date=history_window["timestamps"].iloc[-1],
                        baseline_close=float(history_window["close"].iloc[-1]),
                        x_df=history_window[[*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]].reset_index(drop=True),
                        x_timestamp=history_window["timestamps"].reset_index(drop=True),
                        y_timestamp=actual_future["timestamps"].reset_index(drop=True),
                        actual_future_df=actual_future[["timestamps", *REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]].reset_index(drop=True),
                    )
                )

    return samples_by_context

