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
        verbose: bool = False,
    ) -> None:
        self._service = service
        self._pred_len = pred_len
        self._temperature = temperature
        self._top_p = top_p
        self._sample_count = sample_count
        self._success_mape_threshold = success_mape_threshold
        self._verbose = verbose

    def run(self, samples_by_context: dict[int, list]) -> tuple[list[BacktestEvaluationResult], pd.DataFrame, pd.DataFrame]:
        results: list[BacktestEvaluationResult] = []

        for context_length, samples in samples_by_context.items():
            if not samples:
                logger.info("skip context_length=%s because no valid samples", context_length)
                continue

            logger.info("running backtest for context_length=%s sample_count=%s", context_length, len(samples))
            for sample in samples:
                sample, prediction_frame = self._service.predict(
                    prepared_series=sample,
                    pred_len=self._pred_len,
                    temperature=self._temperature,
                    top_p=self._top_p,
                    sample_count=self._sample_count,
                    verbose=self._verbose,
                )
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
