"""Decoder benchmark under biased circuit-level noise.

Real hardware often dephases far more than it bit-flips. This experiment sweeps
the noise bias eta = p_Z / (p_X + p_Y) from depolarizing (eta = 0.5) to heavily
Z-biased (eta = 100) and asks: **does the decoder ranking measured under
depolarizing noise survive when the noise is structured?**

Design choices that matter for interpreting the results:

- The benchmark runs on the **X-basis memory**: under Z bias its logical
  observable is the stressed one (it fails via Z errors), so this is the
  direction where decoder differences have consequences.
- All decoders consume the *same* detector error models, whose priors reflect
  the bias — so every decoder is given the same, correct soft information.
- A **validation stage** first refits the threshold at eta = 0.5 (exactly
  depolarizing single-qubit channels) with MWPM. The two-qubit noise convention
  here (independent biased channels per qubit) differs from DEPOLARIZE2, so the
  within-model baseline is refit rather than borrowed from the depolarizing
  study.

Run from the repository root with the project venv:

    ./.venv/Scripts/python.exe -u experiments/biased_noise_benchmark.py
"""

from __future__ import annotations

import argparse
import functools
import math
from pathlib import Path

import numpy as np

from qec.decoders import DECODER_LABELS, get_custom_decoders
from qec.harness import (
    build_threshold_tasks,
    collect_statistics,
    estimate_threshold,
    plot_threshold,
)
from qec.noise import build_biased_surface_code_memory_circuit

# --- Validation stage (eta = 0.5, Z-basis memory, MWPM) ----------------------
VALIDATION_DISTANCES = (3, 5, 7)
VALIDATION_ERROR_RATES = tuple(np.geomspace(0.002, 0.015, 8).round(6))

# --- Main stage: bias sweep ---------------------------------------------------
BIAS_VALUES = (0.5, 3.0, 10.0, 30.0, 100.0)
MAIN_DISTANCE = 5
MAIN_ERROR_RATES = tuple(np.geomspace(0.002, 0.012, 6).round(6))
HEADLINE_P = 0.005  # operating point for the p_L-vs-eta headline figure

DECODER_STYLE = {
    "mwpm": {"color": "#2a78d6", "marker": "o"},
    "unionfind": {"color": "#1baf7a", "marker": "s"},
    "bposd": {"color": "#eda100", "marker": "^"},
}

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_FIGURE = REPO_ROOT / "figures" / "biased_validation_threshold.png"
SWEEP_FIGURE = REPO_ROOT / "figures" / "biased_noise_decoders.png"
HEADLINE_FIGURE = REPO_ROOT / "figures" / "biased_noise_headline.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-shots", type=int, default=25_000)
    parser.add_argument("--max-errors", type=int, default=250)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the eta = 0.5 threshold refit stage.")
    args = parser.parse_args()

    if not args.skip_validation:
        _run_validation(args)
    _run_bias_sweep(args)


def _run_validation(args) -> None:
    """Refit the depolarizing-limit threshold within the biased noise model."""
    print("=== Validation: eta = 0.5 (depolarizing limit), memory_z, MWPM ===")
    builder = functools.partial(
        build_biased_surface_code_memory_circuit, bias_eta=0.5, basis="z"
    )
    tasks = build_threshold_tasks(
        builder,
        distances=VALIDATION_DISTANCES,
        physical_error_rates=VALIDATION_ERROR_RATES,
        rounds=lambda d: d,
        extra_metadata={"code": "rotated_surface", "eta": 0.5, "basis": "z"},
    )
    stats = collect_statistics(
        tasks,
        max_shots=2 * args.max_shots,
        max_errors=args.max_errors,
        num_workers=args.workers,
    )
    estimate = None
    try:
        estimate = estimate_threshold(stats)
        print(f"\nWithin-model depolarizing baseline:  {estimate}")
        if not 0.003 <= estimate.p_th <= 0.012:
            print("  WARNING: eta = 0.5 threshold far from the depolarizing "
                  "band — the noise transformation may be mis-wired.")
    except ValueError as exc:
        print(f"\nValidation fit failed: {exc}")

    _with_agg_figure(VALIDATION_FIGURE, figsize=(7, 5), render=lambda fig, ax: (
        plot_threshold(
            ax, stats,
            title="Biased-noise model at η = 0.5 — depolarizing-limit check",
            mark_threshold=estimate,
        )
    ))


def _run_bias_sweep(args) -> None:
    """The main experiment: decoders x bias grid on the stressed memory."""
    print(f"\n=== Bias sweep: memory_x, d = {MAIN_DISTANCE}, "
          f"eta in {BIAS_VALUES} ===")
    decoders = get_custom_decoders()

    tasks = []
    for eta in BIAS_VALUES:
        builder = functools.partial(
            build_biased_surface_code_memory_circuit, bias_eta=eta, basis="x"
        )
        tasks.extend(build_threshold_tasks(
            builder,
            distances=[MAIN_DISTANCE],
            physical_error_rates=MAIN_ERROR_RATES,
            rounds=lambda d: d,
            extra_metadata={"code": "rotated_surface", "eta": eta, "basis": "x"},
        ))
    print(f"{len(tasks)} tasks x {len(decoders)} decoders...")

    stats = collect_statistics(
        tasks,
        decoders=tuple(decoders),
        custom_decoders=decoders,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        num_workers=args.workers,
    )

    _print_sweep_table(stats)
    _save_sweep_figure(stats)
    _save_headline_figure(stats)


def _rate_and_err(stat) -> tuple[float, float]:
    rate = stat.errors / stat.shots
    return rate, math.sqrt(max(rate * (1 - rate), 1e-12) / stat.shots)


def _print_sweep_table(stats) -> None:
    p_ref = min(MAIN_ERROR_RATES, key=lambda p: abs(p - HEADLINE_P))
    print(f"\nLogical error rate at p = {p_ref}, d = {MAIN_DISTANCE} (memory_x):")
    print(f"  {'eta':>6}  " + "".join(f"{DECODER_LABELS[n]:>28}" for n in DECODER_STYLE))
    for eta in BIAS_VALUES:
        row = [f"  {eta:>6}"]
        for name in DECODER_STYLE:
            match = [s for s in stats
                     if s.decoder == name and s.json_metadata["eta"] == eta
                     and s.json_metadata["p"] == p_ref and s.shots]
            rate = match[0].errors / match[0].shots if match else float("nan")
            row.append(f"{rate:>28.3e}")
        print("".join(row))
    print()


def _save_sweep_figure(stats) -> None:
    def render(fig, axes):
        for ax, eta in zip(axes, BIAS_VALUES):
            for name, style in DECODER_STYLE.items():
                pts = sorted(
                    (s for s in stats
                     if s.decoder == name and s.json_metadata["eta"] == eta
                     and s.shots),
                    key=lambda s: s.json_metadata["p"],
                )
                ps = [s.json_metadata["p"] for s in pts]
                rates, errs = zip(*(_rate_and_err(s) for s in pts))
                ax.errorbar(ps, rates, yerr=errs,
                            color=style["color"], marker=style["marker"],
                            markersize=5, linewidth=2, capsize=2,
                            label=DECODER_LABELS[name])
            label = "0.5 (depolarizing)" if eta == 0.5 else f"{eta:g}"
            ax.set_title(f"η = {label}", fontsize=10)
            ax.set_xlabel("physical error rate $p$")
            ax.loglog()
            ax.grid(which="both", alpha=0.25)
        axes[0].set_ylabel("logical error rate per shot")
        axes[0].legend(fontsize=7)
        fig.suptitle(
            f"Decoders vs noise bias — rotated surface code, d = {MAIN_DISTANCE}, "
            "X-basis memory (stressed direction)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SWEEP_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(BIAS_VALUES), figsize=(16, 3.8),
                             sharey=True, sharex=True)
    try:
        render(fig, axes)
        fig.tight_layout()
        fig.savefig(SWEEP_FIGURE, dpi=150)
        print(f"Saved {SWEEP_FIGURE}")
    finally:
        plt.close(fig)


def _save_headline_figure(stats) -> None:
    p_ref = min(MAIN_ERROR_RATES, key=lambda p: abs(p - HEADLINE_P))

    def render(fig, ax):
        for name, style in DECODER_STYLE.items():
            etas, rates, errs = [], [], []
            for eta in BIAS_VALUES:
                match = [s for s in stats
                         if s.decoder == name and s.json_metadata["eta"] == eta
                         and s.json_metadata["p"] == p_ref and s.shots]
                if match:
                    r, e = _rate_and_err(match[0])
                    etas.append(eta); rates.append(r); errs.append(e)
            ax.errorbar(etas, rates, yerr=errs,
                        color=style["color"], marker=style["marker"],
                        markersize=7, linewidth=2, capsize=3,
                        label=DECODER_LABELS[name])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("noise bias  η = $p_Z / (p_X + p_Y)$")
        ax.set_ylabel("logical error rate per shot")
        ax.set_title(f"Decoder accuracy vs noise bias  "
                     f"(d = {MAIN_DISTANCE}, p = {p_ref}, X-basis memory)")
        ax.axvline(0.5, color="#888888", linestyle=":", linewidth=1)
        ax.annotate("depolarizing", (0.5, ax.get_ylim()[0]),
                    textcoords="offset points", xytext=(4, 6),
                    fontsize=8, color="#555555")
        ax.grid(which="both", alpha=0.25)
        ax.legend(fontsize=8)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HEADLINE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        render(fig, ax)
        fig.tight_layout()
        fig.savefig(HEADLINE_FIGURE, dpi=150)
        print(f"Saved {HEADLINE_FIGURE}")
    finally:
        plt.close(fig)


def _with_agg_figure(path: Path, *, figsize, render) -> None:
    """Save a single-axes figure via the headless Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    try:
        render(fig, ax)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
