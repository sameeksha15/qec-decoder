"""Decoder benchmark: MWPM vs union-find vs BP+OSD on the surface code.

Two questions, two figures:

1. **Accuracy** — logical error rate vs physical error rate for each decoder,
   at d = 3, 5, 7 (one panel per distance), on identically sampled circuits.
2. **Latency** — median per-shot decode time vs distance at a fixed operating
   point below threshold, every decoder timed on the same syndrome batch
   through the same bit-packed API (see ``qec/harness/latency.py`` for the
   methodology).

Expected qualitative outcome (to be tested, not assumed): MWPM is near-optimal
and fast on surface codes; union-find trades a little accuracy for speed;
BP+OSD is the most general but pays in wall-clock time. A follow-up study asks
whether that ranking survives realistic (biased/correlated) noise.

Run from the repository root with the project venv:

    ./.venv/Scripts/python.exe -u experiments/decoder_benchmark.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from qec.codes import build_surface_code_memory_circuit
from qec.decoders import DECODER_LABELS, get_custom_decoders
from qec.harness import (
    benchmark_decoder_latency,
    build_threshold_tasks,
    collect_statistics,
)

# Matplotlib imported lazily in the figure helpers (kept out of worker imports).

DISTANCES = (3, 5, 7)
ERROR_RATES = tuple(np.geomspace(0.002, 0.012, 7).round(6))

# Fixed operating point for the latency study: below threshold (real machines
# must operate there), but high enough that syndromes are non-trivially dense.
LATENCY_P = 0.005

# Validated categorical palette (dataviz slots 1-3) + distinct markers as
# secondary encoding, so decoder identity never rides on color alone.
DECODER_STYLE = {
    "mwpm": {"color": "#2a78d6", "marker": "o"},
    "unionfind": {"color": "#1baf7a", "marker": "s"},
    "bposd": {"color": "#eda100", "marker": "^"},
}

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCURACY_FIGURE = REPO_ROOT / "figures" / "decoder_accuracy.png"
LATENCY_FIGURE = REPO_ROOT / "figures" / "decoder_latency.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-shots", type=int, default=25_000)
    parser.add_argument("--max-errors", type=int, default=250)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--latency-shots", type=int, default=2_000)
    args = parser.parse_args()

    decoders = get_custom_decoders()

    # --- Accuracy: same tasks, three decoders --------------------------------
    tasks = build_threshold_tasks(
        build_surface_code_memory_circuit,
        distances=DISTANCES,
        physical_error_rates=ERROR_RATES,
        rounds=lambda d: d,
        extra_metadata={"code": "rotated_surface"},
    )
    print(f"Accuracy: {len(tasks)} tasks x {len(decoders)} decoders...")
    stats = collect_statistics(
        tasks,
        decoders=tuple(decoders),
        custom_decoders=decoders,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        num_workers=args.workers,
    )

    # --- Latency: identical syndrome batches at the operating point ----------
    print(f"\nLatency at p = {LATENCY_P} ({args.latency_shots} shots/batch)...")
    latency_by_distance: dict[int, list] = {}
    for d in DISTANCES:
        circuit = build_surface_code_memory_circuit(
            distance=d, rounds=d, physical_error_rate=LATENCY_P
        )
        latency_by_distance[d] = benchmark_decoder_latency(
            circuit, decoders, num_shots=args.latency_shots
        )
        for r in latency_by_distance[d]:
            print(f"  d={d}  {r.decoder_name:<10} {r.us_per_shot:>10.1f} us/shot")

    _print_summary(stats, latency_by_distance)
    _save_accuracy_figure(stats)
    _save_latency_figure(latency_by_distance)


def _stat_key(stat) -> tuple[str, int, float]:
    return stat.decoder, int(stat.json_metadata["d"]), float(stat.json_metadata["p"])


def _print_summary(stats, latency_by_distance) -> None:
    """Accuracy x latency summary at the operating point nearest LATENCY_P."""
    p_ref = min(ERROR_RATES, key=lambda p: abs(p - LATENCY_P))
    print(f"\nSummary at p = {p_ref} (accuracy) / {LATENCY_P} (latency):")
    print(f"  {'decoder':<12} {'d':>3} {'logical_error_rate':>20} {'us/shot':>10}")
    print("  " + "-" * 49)
    for d in DISTANCES:
        lat = {r.decoder_name: r.us_per_shot for r in latency_by_distance[d]}
        for stat in sorted(stats, key=_stat_key):
            name, sd, p = _stat_key(stat)
            if sd == d and p == p_ref and stat.shots:
                rate = stat.errors / stat.shots
                print(f"  {name:<12} {d:>3} {rate:>20.3e} {lat[name]:>10.1f}")
    print()


def _save_accuracy_figure(stats) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ACCURACY_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(DISTANCES), figsize=(12, 4.2), sharey=True, sharex=True
    )
    try:
        for ax, d in zip(axes, DISTANCES):
            for name, style in DECODER_STYLE.items():
                pts = sorted(
                    (s for s in stats
                     if s.decoder == name and s.json_metadata["d"] == d and s.shots),
                    key=lambda s: s.json_metadata["p"],
                )
                ps = [s.json_metadata["p"] for s in pts]
                rates = [s.errors / s.shots for s in pts]
                errs = [
                    math.sqrt(max(r * (1 - r), 1e-12) / s.shots)
                    for r, s in zip(rates, pts)
                ]
                ax.errorbar(
                    ps, rates, yerr=errs,
                    color=style["color"], marker=style["marker"],
                    markersize=5, linewidth=2, capsize=2,
                    label=DECODER_LABELS[name],
                )
            ax.set_title(f"d = {d}")
            ax.set_xlabel("physical error rate $p$")
            ax.loglog()
            ax.grid(which="both", alpha=0.25)
        axes[0].set_ylabel("logical error rate per shot")
        axes[0].legend(fontsize=8)
        fig.suptitle("Decoder accuracy on the rotated surface code "
                     "(circuit-level depolarizing noise)")
        fig.tight_layout()
        fig.savefig(ACCURACY_FIGURE, dpi=150)
        print(f"Saved {ACCURACY_FIGURE}")
    finally:
        plt.close(fig)


def _save_latency_figure(latency_by_distance) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    LATENCY_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        for name, style in DECODER_STYLE.items():
            ds = list(latency_by_distance)
            times = [
                next(r.us_per_shot for r in latency_by_distance[d]
                     if r.decoder_name == name)
                for d in ds
            ]
            ax.plot(
                ds, times,
                color=style["color"], marker=style["marker"],
                markersize=7, linewidth=2, label=DECODER_LABELS[name],
            )
            # Direct label at the line's end (relief for low-contrast hues).
            ax.annotate(
                f"{times[-1]:.0f} µs", (ds[-1], times[-1]),
                textcoords="offset points", xytext=(8, 0),
                fontsize=8, color="#333333", va="center",
            )
        ax.set_yscale("log")
        ax.set_xticks(list(latency_by_distance))
        ax.set_xlabel("code distance $d$")
        ax.set_ylabel("decode time per shot (µs, median)")
        ax.set_title(f"Decoder latency at p = {LATENCY_P} (same syndrome batches)")
        ax.grid(which="both", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(LATENCY_FIGURE, dpi=150)
        print(f"Saved {LATENCY_FIGURE}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
