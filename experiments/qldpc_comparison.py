"""qLDPC vs surface code: protection per qubit at code-capacity level.

The bivariate-bicycle (BB) family (Bravyi et al., Nature 2024) encodes many
logical qubits per block at a far higher rate than surface-code patches. This
experiment makes that concrete at the smallest standard instance: the
**[[72, 12, 6]]** BB code — 12 logical qubits in 72 data qubits — against
**12 independent rotated surface-code patches**, which is how a surface-code
architecture would deliver the same logical count:

| block                  | data qubits | rate  | distance |
| ---------------------- | ----------- | ----- | -------- |
| BB [[72, 12, 6]]       | 72          | 0.167 | 6        |
| 12 x surface [[9,1,3]] | 108         | 0.111 | 3        |
| 12 x surface [[25,1,5]]| 300         | 0.040 | 5        |

Everything runs through the identical pipeline: the same code-capacity
circuit builder (perfect MPP syndrome extraction, one iid depolarizing layer),
the same BP+OSD decoder (matching cannot decode BB's non-graph-like checks —
this is exactly where BP+OSD's generality pays), the same sampling harness.
The failure metric is per **block of 12 logical qubits**: for the BB code a
shot fails if any of its 12 observables flip; for the surface patches the
per-patch failure rate q is converted to a block rate 1 - (1 - q)^12.

Code capacity is an idealized model (no faulty syndrome extraction), so these
numbers are *not* comparable to the circuit-level results elsewhere in the
repo; the comparison between codes under the same idealized model is the
point. Run:

    ./.venv/Scripts/python.exe -u experiments/qldpc_comparison.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import sinter

from qec.codes import bb_72_12_6, build_css_code_capacity_circuit, rotated_surface_css_code
from qec.decoders import get_custom_decoders
from qec.harness import collect_statistics

ERROR_RATES = tuple(np.geomspace(0.004, 0.06, 8).round(6))
SURFACE_DISTANCES = (3, 5)
LOGICALS_PER_BLOCK = 12

# Categorical palette (validated), fixed order; markers as secondary encoding.
SERIES_STYLE = {
    "bb": {"color": "#2a78d6", "marker": "o", "label": "BB [[72, 12, 6]] — 72 qubits"},
    "surface_d3": {"color": "#1baf7a", "marker": "s",
                   "label": "12 × surface [[9, 1, 3]] — 108 qubits"},
    "surface_d5": {"color": "#eda100", "marker": "^",
                   "label": "12 × surface [[25, 1, 5]] — 300 qubits"},
}

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_PATH = REPO_ROOT / "figures" / "qldpc_comparison.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-shots", type=int, default=50_000)
    parser.add_argument("--max-errors", type=int, default=400)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    codes = {"bb": bb_72_12_6()}
    for d in SURFACE_DISTANCES:
        codes[f"surface_d{d}"] = rotated_surface_css_code(d)

    tasks = []
    for key, code in codes.items():
        for p in ERROR_RATES:
            circuit = build_css_code_capacity_circuit(
                code, physical_error_rate=p
            )
            tasks.append(sinter.Task(
                circuit=circuit,
                # BB checks are hyperedges: build the DEM without decomposition.
                detector_error_model=circuit.detector_error_model(
                    decompose_errors=False
                ),
                json_metadata={"series": key, "p": p, "n": code.num_qubits,
                               "k": code.num_logicals},
            ))
    print(f"{len(tasks)} tasks (3 blocks x {len(ERROR_RATES)} rates), BP+OSD...")

    stats = collect_statistics(
        tasks,
        decoders=("bposd",),
        custom_decoders=get_custom_decoders(),
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        num_workers=args.workers,
    )

    block_rates = _block_failure_rates(stats)
    _print_table(block_rates)
    _save_figure(block_rates)


def _block_failure_rates(stats) -> dict[str, list[tuple[float, float, float]]]:
    """Per-series (p, block failure rate, standard error) points.

    The BB block's 12 logicals are sampled jointly (any-flip failure); each
    surface patch carries one logical, so its per-shot rate q becomes a
    12-logical block rate 1 - (1 - q)^12, with the standard error propagated
    through the derivative 12 * (1 - q)^11.
    """
    series: dict[str, list[tuple[float, float, float]]] = {k: [] for k in SERIES_STYLE}
    for stat in sorted(stats, key=lambda s: s.json_metadata["p"]):
        if not stat.shots:
            continue
        key = stat.json_metadata["series"]
        q = stat.errors / stat.shots
        q_err = math.sqrt(max(q * (1 - q), 1e-12) / stat.shots)
        if key == "bb":
            rate, err = q, q_err
        else:
            rate = 1.0 - (1.0 - q) ** LOGICALS_PER_BLOCK
            err = LOGICALS_PER_BLOCK * (1.0 - q) ** (LOGICALS_PER_BLOCK - 1) * q_err
        series[key].append((stat.json_metadata["p"], rate, err))
    return series


def _print_table(block_rates) -> None:
    print("\nBlock (12 logical qubits) failure rate per shot, BP+OSD:")
    print(f"  {'p':>9}" + "".join(f"{SERIES_STYLE[k]['label']:>42}" for k in SERIES_STYLE))
    rows = {p: {} for p, *_ in block_rates["bb"]}
    for key, points in block_rates.items():
        for p, rate, _ in points:
            rows.setdefault(p, {})[key] = rate
    for p in sorted(rows):
        cells = "".join(f"{rows[p].get(k, float('nan')):>42.3e}" for k in SERIES_STYLE)
        print(f"  {p:>9.4f}{cells}")
    print()


def _save_figure(block_rates) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: figures are saved, never displayed
    import matplotlib.pyplot as plt

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    try:
        for key, style in SERIES_STYLE.items():
            # Zero-failure points have no log-scale representation; the table
            # reports them as upper bounds instead.
            points = [pt for pt in block_rates[key] if pt[1] > 0]
            if not points:
                continue
            ps, rates, errs = zip(*points)
            ax.errorbar(ps, rates, yerr=errs,
                        color=style["color"], marker=style["marker"],
                        markersize=6, linewidth=2, capsize=2,
                        label=style["label"])
        ax.loglog()
        ax.set_xlabel("physical error rate $p$ (code capacity, depolarizing)")
        ax.set_ylabel("failure rate per shot, block of 12 logical qubits")
        ax.set_title("qLDPC vs surface code at equal logical count\n"
                     "(perfect syndrome extraction, BP+OSD for all)")
        ax.grid(which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIGURE_PATH, dpi=150)
        print(f"Saved {FIGURE_PATH}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
