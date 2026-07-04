"""The rotated-surface-code threshold study.

This is the *credibility* milestone. We reproduce the well-known surface-code
threshold (~0.5-1% under circuit-level depolarizing noise, decoded with MWPM)
from scratch, and extract it quantitatively with a finite-size
scaling fit rather than eyeballing the crossing.

The repetition-code harness is reused unchanged; only the ``code_builder`` differs
(surface code instead of repetition code). That reuse is the point of the
harness's code-agnostic design.

Run from the repository root with the project venv:

    ./.venv/Scripts/python.exe -u experiments/surface_threshold.py

The figure is written to ``figures/surface_threshold.png`` and the fitted
threshold is printed to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qec.codes import build_surface_code_memory_circuit
from qec.harness import (
    build_threshold_tasks,
    collect_statistics,
    estimate_threshold,
    plot_threshold,
)

# Matplotlib is imported lazily inside `_save_threshold_figure` (keeps it
# out of the multiprocessing workers' import path).

# Odd distances, well separated so the curve crossing is legible.
DEFAULT_DISTANCES = (3, 5, 7)

# Log-spaced sweep straddling the expected circuit-level threshold (~0.6%).
# We bias the range to sit around the crossing so the scaling fit has enough
# points in its valid region.
DEFAULT_ERROR_RATES = tuple(np.geomspace(0.001, 0.02, 12).round(6))

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = REPO_ROOT / "figures" / "surface_threshold.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-shots", type=int, default=200_000,
                        help="Hard cap on shots per (distance, p) point.")
    parser.add_argument("--max-errors", type=int, default=500,
                        help="Target logical errors per point before stopping.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel worker processes (default: memory-aware).")
    args = parser.parse_args()

    tasks = build_threshold_tasks(
        build_surface_code_memory_circuit,
        distances=DEFAULT_DISTANCES,
        physical_error_rates=DEFAULT_ERROR_RATES,
        rounds=lambda d: d,  # depth scales with the code distance
        extra_metadata={"code": "rotated_surface"},
    )
    print(f"Built {len(tasks)} tasks "
          f"({len(DEFAULT_DISTANCES)} distances x {len(DEFAULT_ERROR_RATES)} rates).")

    stats = collect_statistics(
        tasks,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        num_workers=args.workers,
    )

    _report_table(stats)

    estimate = None
    try:
        estimate = estimate_threshold(stats)
        print(f"\nFinite-size scaling fit:  {estimate}")
        if not 0.001 <= estimate.p_th <= 0.02:
            print("  WARNING: fitted threshold is outside the expected "
                  "circuit-level band (~0.5-1%); inspect the data.")
    except ValueError as exc:
        print(f"\nThreshold fit skipped: {exc}")

    _save_threshold_figure(stats, estimate)


def _save_threshold_figure(stats, estimate) -> None:
    """Render and save the threshold plot (Matplotlib imported lazily here)."""
    import matplotlib

    matplotlib.use("Agg")  # headless; we only save figures, never display them
    import matplotlib.pyplot as plt

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        plot_threshold(
            ax, stats,
            title="Rotated surface code — circuit-level threshold",
            mark_threshold=estimate,
        )
        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=150)
        print(f"Saved figure to {FIGURE_PATH}")
    finally:
        plt.close(fig)


def _report_table(stats) -> None:
    """Print logical error rate per (distance, p) as a quick sanity table."""
    print("\n  d      p          shots    errors   logical_error_rate")
    print("  " + "-" * 54)
    for stat in sorted(stats, key=lambda s: (s.json_metadata["d"], s.json_metadata["p"])):
        d = stat.json_metadata["d"]
        p = stat.json_metadata["p"]
        rate = stat.errors / stat.shots if stat.shots else float("nan")
        print(f"  {d:<3}  {p:<9.6f}  {stat.shots:>8}  {stat.errors:>6}   {rate:.3e}")
    print()


if __name__ == "__main__":
    main()
