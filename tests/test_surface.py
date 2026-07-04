"""Surface-code tests: circuit constructor and the threshold estimator.

Fast checks (sub-second): verify the surface-code circuit is well formed and that
the finite-size scaling fit recovers a *known* threshold from synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import stim

from qec.codes import build_surface_code_memory_circuit
from qec.harness import estimate_threshold


def test_surface_circuit_has_detectors_and_single_observable() -> None:
    circuit = build_surface_code_memory_circuit(distance=3, rounds=3, physical_error_rate=0.005)
    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_detectors > 0
    assert circuit.num_observables == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance": 4, "rounds": 3, "physical_error_rate": 0.005},   # even distance
        {"distance": 1, "rounds": 3, "physical_error_rate": 0.005},   # too small
        {"distance": 3, "rounds": 0, "physical_error_rate": 0.005},   # no rounds
        {"distance": 3, "rounds": 3, "physical_error_rate": -0.1},    # p out of range
    ],
)
def test_surface_circuit_rejects_bad_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        build_surface_code_memory_circuit(**kwargs)


def test_surface_circuit_compiles_to_decodable_error_model() -> None:
    circuit = build_surface_code_memory_circuit(distance=5, rounds=5, physical_error_rate=0.004)
    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem.num_detectors > 0


@dataclass
class _FakeStat:
    """Minimal stand-in for sinter.TaskStats (duck-typed for the estimator)."""

    shots: int
    errors: int
    json_metadata: dict[str, Any] = field(default_factory=dict)


def test_estimate_threshold_recovers_known_value() -> None:
    # Generate synthetic data that obeys the scaling ansatz exactly:
    #   p_L = A + B*(p - p_th)*d**(1/nu)   (linear; all curves cross at p_th).
    true_p_th, true_nu, a, b = 0.006, 1.5, 0.2, 6.0
    shots = 200_000
    stats: list[_FakeStat] = []
    for d in (3, 5, 7):
        for p in (0.004, 0.005, 0.006, 0.007, 0.008):
            x = (p - true_p_th) * d ** (1.0 / true_nu)
            p_l = a + b * x
            errors = round(p_l * shots)
            stats.append(_FakeStat(shots, errors, {"d": d, "p": p}))

    est = estimate_threshold(stats, p_window=(0.003, 0.009))
    assert est.p_th == pytest.approx(true_p_th, abs=2e-4)
    assert est.num_points == 15


def test_estimate_threshold_rejects_too_few_points() -> None:
    stats = [_FakeStat(1000, 100, {"d": 3, "p": 0.006})]
    with pytest.raises(ValueError):
        estimate_threshold(stats, p_window=(0.0, 1.0))
