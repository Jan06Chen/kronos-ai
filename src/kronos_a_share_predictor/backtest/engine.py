from __future__ import annotations

import logging

import pandas as pd

from kronos_a_share_predictor.backtest.metrics import (
    BacktestEvaluationResult,
    evaluate_backtest_prediction,
    results_to_detail_frame,
    summarize_results,
)


logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(
        self,
        service,
        pred_len: int,
        temperature: float,
        top_p: float,
        sample_count: int,
        success_mape_threshold: float,
        batch_size: int,
    ) -> None:
        self._service = service
        self._pred_len = pred_len
        self._temperature = temperature
        self._top_p = top_p
        self._sample_count = sample_count
        self._success_mape_threshold = success_mape_threshold
        self._batch_size = max(batch_size, 1)

    def run(self, samples_by_context: dict[int, list]) -> tuple[list[BacktestEvaluationResult], pd.DataFrame, pd.DataFrame]:
        results: list[BacktestEvaluationResult] = []

        for context_length, samples in samples_by_context.items():
            if not samples:
                logger.info("skip context_length=%s because no valid samples", context_length)
                continue

            logger.info("running backtest for context_length=%s sample_count=%s", context_length, len(samples))
            for start_index in range(0, len(samples), self._batch_size):
                batch = samples[start_index : start_index + self._batch_size]
                predictions = self._service.predict_batch(
                    prepared_series=batch,
                    pred_len=self._pred_len,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    sample_count=self._sample_count,
                )
                for sample, prediction_frame in predictions:
                    results.append(
                        evaluate_backtest_prediction(
                            sample=sample,
                            prediction_frame=prediction_frame,
                            success_mape_threshold=self._success_mape_threshold,
                        )
                    )

        detail_frame = results_to_detail_frame(results)
        summary_frame = summarize_results(results, tuple(samples_by_context.keys()))
        return results, summary_frame, detail_frame
