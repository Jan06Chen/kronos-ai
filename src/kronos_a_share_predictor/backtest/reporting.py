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


def write_sampling_tuning_reports(
    summary_frame: pd.DataFrame,
    best_detail_frame: pd.DataFrame,
    output_dir: Path,
    run_uuid: str,
) -> tuple[str | None, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"sampling_tuning_{run_uuid}_summary.csv"
    detail_path = output_dir / f"sampling_tuning_{run_uuid}_best_details.csv"

    summary_frame.to_csv(summary_path, index=False)
    best_detail_frame.to_csv(detail_path, index=False)
    return str(summary_path), str(detail_path)


def _sampling_paths(output_dir: Path, run_uuid: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"sampling_tuning_{run_uuid}_summary.csv"
    details_path = output_dir / f"sampling_tuning_{run_uuid}_details.csv"
    best_path = output_dir / f"sampling_tuning_{run_uuid}_best_details.csv"
    return summary_path, details_path, best_path


def append_sampling_summary_row(record: dict, output_dir: Path, run_uuid: str) -> str:
    summary_path, _, _ = _sampling_paths(output_dir, run_uuid)
    df = pd.DataFrame([record])
    if summary_path.exists():
        df.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        df.to_csv(summary_path, index=False)
    return str(summary_path)


def append_sampling_details_frame(detail_frame: pd.DataFrame, output_dir: Path, run_uuid: str) -> str:
    _, details_path, _ = _sampling_paths(output_dir, run_uuid)
    if detail_frame.empty:
        return str(details_path)
    if details_path.exists():
        detail_frame.to_csv(details_path, mode="a", header=False, index=False)
    else:
        detail_frame.to_csv(details_path, index=False)
    return str(details_path)


def write_best_detail_file(best_detail_frame: pd.DataFrame, output_dir: Path, run_uuid: str) -> str:
    _, _, best_path = _sampling_paths(output_dir, run_uuid)
    best_detail_frame.to_csv(best_path, index=False)
    return str(best_path)

