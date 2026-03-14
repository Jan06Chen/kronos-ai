from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_backtest_reports(summary_frame: pd.DataFrame, detail_frame: pd.DataFrame, output_dir: Path, run_uuid: str) -> tuple[str | None, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"backtest_{run_uuid}_summary.csv"
    detail_path = output_dir / f"backtest_{run_uuid}_details.csv"

    summary_frame.to_csv(summary_path, index=False)
    detail_frame.to_csv(detail_path, index=False)
    return str(summary_path), str(detail_path)
