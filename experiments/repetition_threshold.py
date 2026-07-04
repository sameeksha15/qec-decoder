"""Harness validation: a repetition-code threshold plot.

This is the harness shake-down. We are not after a research result here — we want
to confirm that the Stim -> Sinter -> PyMatching pipeline produces the textbook
qualitative behaviour: below threshold, larger code distance gives a *lower*
logical error rate, so the per-distance curves cross at the threshold.

Run from the repository root with the project venv:

    ./.venv/Scripts/python.exe experiments/repetition_threshold.py

The figure is written to ``figures/repetition_threshold.png`` and the
crossing region is printed to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qec.codes import build_repetition_memory_circuit
from qec.harness import build_threshold_tasks, collect_statistics, plot_threshold

# NOTE: Matplotlib is imported lazily inside `_save_threshold_figure`, not at
# module top level. On Windows each multiprocessing worker re-imports this
# module; keeping Matplotlib out of the import path stops every worker from
# needlessly loading it (which previously exhausted RAM and crashed the run).

# Distances chosen odd and well separated so the curve crossing is legible.
DEFAULT_DISTANCES = (3, 5, 7, 9)

# Log-spaced sweep straddling the expected repetition-code threshold so we
# capture both the suppressing (below) and amplifying (above) regimes.
DEFAULT_ERROR_RATES = tuple(np.geomspace(0.01, 0.20, 12).round(5))

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = REPO_ROOT / "figures" / "repetition_threshold.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-shots",
        type=int,
        default=200_000,
        help="Hard cap on shots per (distance, p) point.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=500,
        help="Target logical errors per point before sampling stops early.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes (default: CPU count - 1).",
    )
    args = parser.parse_args()

    tasks = build_threshold_tasks(
        build_repetition_memory_circuit,
        distances=DEFAULT_DISTANCES,
        physical_error_rates=DEFAULT_ERROR_RATES,
        rounds=lambda d: d,  # standard convention: depth scales with the code
        extra_metadata={"code": "repetition"},
    )
    print(f"Built {len(tasks)} tasks "
          f"({len(DEFAULT_DISTANCES)} distances x {len(DEFAULT_ERROR_RATES)} rates).")

    stats = collect_statistics(
        tasks,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        num_workers=args.workers,
    )

    _report_crossing(stats)
    _save_threshold_figure(stats)


def _save_threshold_figure(stats) -> None:
    """Render and save the threshold plot (Matplotlib imported lazily here)."""
    # Force the non-interactive Agg backend before pyplot loads: we only save
    # figures, never display them, so we avoid depending on a Tcl/Tk install.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        plot_threshold(ax, stats, title="Repetition code — harness validation")
        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=150)
        print(f"Saved figure to {FIGURE_PATH}")
    finally:
        # Explicitly close to release the backend's buffers — the right
        # "garbage collection" for Matplotlib, far better than a blind
        # gc.collect().
        plt.close(fig)


def _report_crossing(stats) -> None:
    """Print logical error rate per (distance, p) as a quick sanity table."""
    print("\n  d      p        shots    errors   logical_error_rate")
    print("  " + "-" * 52)
    for stat in sorted(stats, key=lambda s: (s.json_metadata["d"], s.json_metadata["p"])):
        d = stat.json_metadata["d"]
        p = stat.json_metadata["p"]
        rate = stat.errors / stat.shots if stat.shots else float("nan")
        print(f"  {d:<3}  {p:<8.5f}  {stat.shots:>8}  {stat.errors:>6}   {rate:.3e}")
    print()


if __name__ == "__main__":
    main()
