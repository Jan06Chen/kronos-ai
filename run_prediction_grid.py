from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a parameter grid for main.py sequentially")
    parser.add_argument("--python-bin", default="python3.11", help="Python executable used to run main.py")
    parser.add_argument("--recommendation-date", default="2026-03-09", help="Recommendation date, format YYYY-MM-DD")
    parser.add_argument("--endtime", default="2026-03-11", help="History end date, format YYYY-MM-DD")
    parser.add_argument("--lookback-days", type=int, default=512, help="Historical lookback days")
    parser.add_argument("--min-history-points", type=int, default=60, help="Base minimum history points; script will cap it to effective context length")
    parser.add_argument("--context-lengths", default="30,60,90,120,240,400", help="Comma-separated context lengths")
    parser.add_argument("--temperatures", default="0.8,0.9,1.1,1.2", help="Comma-separated temperatures")
    parser.add_argument("--top-ps", default="0.8,0.9,1.0,1.1", help="Comma-separated top_p values")
    parser.add_argument("--sample-counts", default="1,3,5", help="Comma-separated sample counts")
    parser.add_argument("--device", help="Optional inference device override")
    parser.add_argument("--verbose-inference", action="store_true", help="Pass verbose inference to main.py")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately when one command fails")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args()


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _build_command(args: argparse.Namespace, *, context_length: int, temperature: float, top_p: float, sample_count: int) -> list[str]:
    effective_context_length = min(context_length, args.lookback_days, 512)
    min_history_points = min(args.min_history_points, effective_context_length)
    command = [
        args.python_bin,
        "main.py",
        "--recommendation-date",
        args.recommendation_date,
        "--endtime",
        args.endtime,
        "--context-length",
        str(context_length),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--sample-count",
        str(sample_count),
        "--lookback-days",
        str(args.lookback_days),
        "--min-history-points",
        str(min_history_points),
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.verbose_inference:
        command.append("--verbose-inference")
    return command


def main() -> int:
    args = _parse_args()
    _validate_positive("lookback-days", args.lookback_days)
    _validate_positive("min-history-points", args.min_history_points)

    context_lengths = _parse_int_list(args.context_lengths)
    temperatures = _parse_float_list(args.temperatures)
    top_ps = _parse_float_list(args.top_ps)
    sample_counts = _parse_int_list(args.sample_counts)

    for value in context_lengths:
        _validate_positive("context-length", value)
    for value in sample_counts:
        _validate_positive("sample-count", value)
    for value in temperatures:
        if value <= 0:
            raise ValueError("temperature must be greater than 0")

    total = 0
    executed = 0
    skipped = 0
    failed = 0

    for context_length, temperature, top_p, sample_count in product(context_lengths, temperatures, top_ps, sample_counts):
        total += 1
        if not 0 < top_p <= 1:
            skipped += 1
            print(
                f"[SKIP] context_length={context_length} temperature={temperature} top_p={top_p} sample_count={sample_count} "
                f"because main.py only accepts top-p in (0, 1]"
            )
            continue

        command = _build_command(
            args,
            context_length=context_length,
            temperature=temperature,
            top_p=top_p,
            sample_count=sample_count,
        )
        print(f"[RUN ] {' '.join(command)}")

        if args.dry_run:
            executed += 1
            continue

        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        if completed.returncode != 0:
            failed += 1
            print(f"[FAIL] exit_code={completed.returncode}")
            if args.stop_on_error:
                break
        else:
            executed += 1
            print("[ OK ]")

    print(
        f"finished total={total} executed={executed} skipped={skipped} failed={failed} dry_run={args.dry_run}"
    )
    return 1 if failed > 0 and args.stop_on_error else 0


if __name__ == "__main__":
    sys.exit(main())