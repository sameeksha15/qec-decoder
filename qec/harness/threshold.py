"""Generic threshold-study harness built on Sinter.

The workflow is always the same regardless of which code we study:

1. Build a grid of noisy circuits over (distance x physical_error_rate).
2. Monte-Carlo sample each with Sinter until enough logical errors accumulate.
3. Plot logical error rate vs physical error rate, grouped by distance.

Below threshold, increasing the distance *suppresses* the logical error rate, so
the per-distance curves fan out and cross near the threshold. Keeping this code
decoupled from any specific code constructor means every new code reuses
it without modification — we just pass a different ``code_builder``.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sinter
import stim

if TYPE_CHECKING:
    # Imported only for type checking so that worker processes (which re-import
    # this module) don't pay to load Matplotlib. Safe because `from __future__
    # import annotations` makes all annotations lazy strings.
    from matplotlib.axes import Axes


def _ensure_worker_interpreter() -> None:
    """Pin multiprocessing workers to *this* venv's Python interpreter.

    On Windows, multiprocessing 'spawn' can launch worker processes under the
    base interpreter (``sys.base_prefix``) instead of the active virtual
    environment. Those workers then lack our QEC packages, fail to import, and
    silently hang the whole sampling run. We saw exactly this on Python 3.13rc1.
    Pinning the executable to the venv interpreter makes the workers inherit our
    installed packages. The call is idempotent and a no-op outside a venv.
    """
    if sys.prefix == sys.base_prefix:  # not running inside a virtual environment
        return
    if sys.platform == "win32":
        candidate = Path(sys.prefix) / "Scripts" / "python.exe"
    else:
        candidate = Path(sys.prefix) / "bin" / "python"
    if candidate.exists():
        multiprocessing.set_executable(str(candidate))


# Each spawned worker re-imports the sampling stack; empirically ~0.25 GB each.
# We cap workers by available RAM so a high core count can't trigger a
# MemoryError mid-run on a memory-constrained machine.
_GIB_PER_WORKER = 0.4


def _default_worker_count() -> int:
    """Pick a worker count bounded by both CPU cores and available memory."""
    cpu_bound = max(1, (os.cpu_count() or 2) - 1)  # leave one core responsive

    free_gib = _available_memory_gib()
    if free_gib is None:
        # Can't introspect memory; stay conservative rather than risk a crash.
        return min(cpu_bound, 6)

    # Reserve ~1 GiB headroom for the parent process and the OS.
    mem_bound = max(1, int((free_gib - 1.0) / _GIB_PER_WORKER))
    return max(1, min(cpu_bound, mem_bound))


def _available_memory_gib() -> float | None:
    """Best-effort free physical memory in GiB, or None if undeterminable."""
    try:  # Windows
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / (1024**3)
    except (OSError, AttributeError, ValueError):
        pass

    try:  # POSIX
        return (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / (1024**3)
    except (OSError, ValueError, AttributeError):
        return None

# A code builder is any callable that turns (distance, rounds, p) into a circuit.
# Both the repetition code and the surface code satisfy this contract.
CodeBuilder = Callable[..., stim.Circuit]


def build_threshold_tasks(
    code_builder: CodeBuilder,
    *,
    distances: Sequence[int],
    physical_error_rates: Sequence[float],
    rounds: int | Callable[[int], int],
    extra_metadata: dict[str, Any] | None = None,
) -> list[sinter.Task]:
    """Build the (distance x error-rate) grid of Sinter tasks.

    Args:
        code_builder: Callable returning a ``stim.Circuit`` given keyword
            arguments ``distance``, ``rounds`` and ``physical_error_rate``.
        distances: Code distances to sweep.
        physical_error_rates: Physical error rates to sweep.
        rounds: Either a fixed round count, or a callable ``d -> rounds`` (the
            usual convention is ``rounds = d`` so the experiment depth scales
            with the code).
        extra_metadata: Optional fields merged into every task's metadata, handy
            for tagging a run (e.g. the noise model or decoder under test).

    Returns:
        A flat list of ``sinter.Task`` objects, one per grid point.
    """
    if not distances:
        raise ValueError("distances must be non-empty")
    if not physical_error_rates:
        raise ValueError("physical_error_rates must be non-empty")

    tasks: list[sinter.Task] = []
    for distance in distances:
        n_rounds = rounds(distance) if callable(rounds) else rounds
        for p in physical_error_rates:
            circuit = code_builder(
                distance=distance,
                rounds=n_rounds,
                physical_error_rate=p,
            )
            metadata: dict[str, Any] = {"d": distance, "p": p, "rounds": n_rounds}
            if extra_metadata:
                metadata.update(extra_metadata)
            tasks.append(sinter.Task(circuit=circuit, json_metadata=metadata))
    return tasks


def collect_statistics(
    tasks: Iterable[sinter.Task],
    *,
    decoders: Sequence[str] = ("pymatching",),
    max_shots: int = 1_000_000,
    max_errors: int = 1_000,
    num_workers: int | None = None,
    print_progress: bool = True,
    custom_decoders: dict[str, sinter.Decoder] | None = None,
) -> list[sinter.TaskStats]:
    """Monte-Carlo sample the tasks in parallel until the stopping rule is met.

    Sampling stops per-task once either ``max_errors`` logical errors are seen
    (enough for a tight confidence interval) or ``max_shots`` is reached
    (a budget cap so deep-below-threshold points don't run forever).

    Args:
        tasks: Tasks produced by :func:`build_threshold_tasks`.
        decoders: Decoder names understood by Sinter, or keys into
            ``custom_decoders`` (the decoder benchmark passes several).
        max_shots: Hard cap on shots per task.
        max_errors: Target number of logical errors per task.
        num_workers: Parallel worker processes; defaults to the CPU count.
        print_progress: Stream Sinter's live progress to stderr.

    Returns:
        One ``sinter.TaskStats`` per task, carrying shot/error counts.
    """
    if num_workers is None:
        num_workers = _default_worker_count()

    _ensure_worker_interpreter()

    return sinter.collect(
        num_workers=num_workers,
        tasks=list(tasks),
        decoders=list(decoders),
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=print_progress,
        custom_decoders=custom_decoders,
    )


@dataclass(frozen=True)
class ThresholdEstimate:
    """Result of a finite-size scaling fit for the threshold."""

    p_th: float          # estimated threshold
    p_th_stderr: float   # 1-sigma uncertainty on the threshold
    nu: float            # critical exponent (controls how the curves collapse)
    nu_stderr: float
    num_points: int      # data points used in the fit
    p_window: tuple[float, float]  # (min, max) physical error rate fitted

    def __str__(self) -> str:
        return (
            f"p_th = {self.p_th:.4%} +/- {self.p_th_stderr:.4%}  "
            f"(nu = {self.nu:.2f} +/- {self.nu_stderr:.2f}, "
            f"{self.num_points} points in "
            f"[{self.p_window[0]:.4f}, {self.p_window[1]:.4f}])"
        )


def estimate_threshold(
    stats: Sequence[sinter.TaskStats],
    *,
    p_window: tuple[float, float] | None = None,
    min_logical_rate: float = 0.01,
    max_logical_rate: float = 0.40,
    pth_guess: float | None = None,
    nu_guess: float = 1.5,
) -> ThresholdEstimate:
    """Extract the threshold via a finite-size scaling collapse.

    Near threshold the logical error rate of every distance collapses onto one
    universal curve when plotted against the rescaled variable
    ``x = (p - p_th) * d**(1/nu)``. We fit the standard quadratic ansatz

        p_L = A + B*x + C*x**2

    jointly across all distances, with ``p_th`` and ``nu`` as shared free
    parameters. This is the accepted way to report a threshold *with* an
    uncertainty, rather than eyeballing where the curves cross.

    The quadratic form is only valid *near* threshold, so points are restricted
    to a fitting window: either an explicit ``p_window``, or (by default) the
    band where the logical error rate lies in
    ``[min_logical_rate, max_logical_rate]`` — which automatically brackets the
    crossing for any code.

    Args:
        stats: Statistics from :func:`collect_statistics`, spanning >= 2
            distances and several physical error rates around the crossing.
        p_window: Optional explicit ``(p_min, p_max)`` fitting window. Overrides
            the logical-rate band selection when given.
        min_logical_rate: Lower edge of the auto rate band (ignored if
            ``p_window`` is set).
        max_logical_rate: Upper edge of the auto rate band.
        pth_guess: Initial guess for the threshold; defaults to the median ``p``
            of the selected points.
        nu_guess: Initial guess for the critical exponent.

    Returns:
        A :class:`ThresholdEstimate`.

    Raises:
        ValueError: If fewer than 5 usable points survive selection (the fit has
            5 free parameters), or if the fit fails to converge.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    ds: list[int] = []
    ps: list[float] = []
    pls: list[float] = []
    sigmas: list[float] = []
    for stat in stats:
        if stat.shots == 0:
            continue
        d = int(stat.json_metadata["d"])
        p = float(stat.json_metadata["p"])
        rate = stat.errors / stat.shots

        if p_window is not None:
            if not (p_window[0] <= p <= p_window[1]):
                continue
        elif not (min_logical_rate <= rate <= max_logical_rate):
            continue

        # Binomial standard error, floored at one "effective count" so that
        # near-zero-error points still carry a finite (large) uncertainty.
        stderr = max(math.sqrt(rate * (1.0 - rate) / stat.shots), 1.0 / stat.shots)
        ds.append(d)
        ps.append(p)
        pls.append(rate)
        sigmas.append(stderr)

    if len(ps) < 5:
        raise ValueError(
            f"need >= 5 points near threshold to fit, got {len(ps)}; "
            "widen the sweep or the fitting window"
        )

    p_arr = np.asarray(ps)
    d_arr = np.asarray(ds, dtype=float)
    pl_arr = np.asarray(pls)
    sigma_arr = np.asarray(sigmas)

    def _collapse(x_data, p_th, nu, a, b, c):
        p, d = x_data
        rescaled = (p - p_th) * np.power(d, 1.0 / nu)
        return a + b * rescaled + c * rescaled**2

    p0 = [
        pth_guess if pth_guess is not None else float(np.median(p_arr)),
        nu_guess,
        float(np.median(pl_arr)),
        0.0,
        0.0,
    ]

    try:
        popt, pcov = curve_fit(
            _collapse,
            (p_arr, d_arr),
            pl_arr,
            p0=p0,
            sigma=sigma_arr,
            absolute_sigma=True,
            maxfev=20000,
        )
    except (RuntimeError, ValueError) as exc:  # non-convergence / bad inputs
        raise ValueError(f"threshold fit did not converge: {exc}") from exc

    perr = np.sqrt(np.diag(pcov))
    return ThresholdEstimate(
        p_th=float(popt[0]),
        p_th_stderr=float(perr[0]),
        nu=float(popt[1]),
        nu_stderr=float(perr[1]),
        num_points=len(ps),
        p_window=(float(p_arr.min()), float(p_arr.max())),
    )


def plot_threshold(
    ax: Axes,
    stats: Sequence[sinter.TaskStats],
    *,
    title: str = "Logical error rate vs physical error rate",
    mark_threshold: ThresholdEstimate | None = None,
) -> Axes:
    """Render a threshold plot onto a Matplotlib axis.

    Uses Sinter's own plotting helper so error bars and per-distance grouping
    follow the conventions reviewers expect to see.

    Args:
        ax: Target axis (created by the caller, who also owns saving/closing it).
        stats: Statistics from :func:`collect_statistics`.
        title: Plot title.
        mark_threshold: If given, draw the fitted threshold as a vertical line
            with a shaded +/- 1-sigma band.

    Returns:
        The same axis, for chaining.
    """
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=lambda stat: f"d = {stat.json_metadata['d']}",
    )
    if mark_threshold is not None:
        ax.axvline(
            mark_threshold.p_th,
            color="k",
            linestyle="--",
            linewidth=1,
            label=f"$p_{{th}}$ = {mark_threshold.p_th:.3%}",
        )
        ax.axvspan(
            mark_threshold.p_th - mark_threshold.p_th_stderr,
            mark_threshold.p_th + mark_threshold.p_th_stderr,
            color="k",
            alpha=0.08,
        )
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate per shot")
    ax.set_title(title)
    ax.loglog()
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    return ax
