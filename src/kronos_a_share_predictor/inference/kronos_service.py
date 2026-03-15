from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd

from kronos_a_share_predictor.data.transformers import PreparedSeries


class KronosService:
    def __init__(
        self,
        repo_path: Path,
        tokenizer_id: str,
        model_id: str,
        max_context: int,
        device: str | None,
    ) -> None:
        self._repo_path = repo_path.resolve()
        self._tokenizer_id = tokenizer_id
        self._model_id = model_id
        self._max_context = max_context
        self._device = device
        self._predictor = self._load_predictor()

    def _load_predictor(self):
        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"Kronos 仓库不存在: {self._repo_path}. 请先执行 git clone https://github.com/shiyu-coder/Kronos.git {self._repo_path}"
            )

        repo_path_str = str(self._repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)

        model_module = import_module("model")
        tokenizer = model_module.KronosTokenizer.from_pretrained(self._tokenizer_id)
        model = model_module.Kronos.from_pretrained(self._model_id)
        tokenizer.eval()
        model.eval()
        return model_module.KronosPredictor(
            model,
            tokenizer,
            device=self._device,
            max_context=self._max_context,
        )

    def predict_batch(
        self,
        prepared_series: list[PreparedSeries] | list[Any],
        pred_len: int,
        temperature: float,
        top_p: float,
        sample_count: int,
        verbose: bool = False,
    ) -> list[tuple[PreparedSeries, pd.DataFrame]]:
        prediction_frames = self._predictor.predict_batch(
            df_list=[item.x_df for item in prepared_series],
            x_timestamp_list=[item.x_timestamp for item in prepared_series],
            y_timestamp_list=[item.y_timestamp for item in prepared_series],
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )
        return list(zip(prepared_series, prediction_frames))

    def predict(
        self,
        prepared_series: PreparedSeries | Any,
        pred_len: int,
        temperature: float,
        top_p: float,
        sample_count: int,
        verbose: bool = False,
    ) -> tuple[PreparedSeries, pd.DataFrame]:
        prediction_frame = self._predictor.predict(
            df=prepared_series.x_df,
            x_timestamp=prepared_series.x_timestamp,
            y_timestamp=prepared_series.y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )
        return prepared_series, prediction_frame
