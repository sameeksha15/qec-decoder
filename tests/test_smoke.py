"""Fast smoke tests: catch wiring/regression breakage without long sampling.

These run in well under a second — they verify the pipeline *connects*, not that
the physics is converged. Run with: ``./.venv/Scripts/python.exe -m pytest``.
"""

from __future__ import annotations

import sinter
import stim
import pytest

from qec.codes import build_repetition_memory_circuit
from qec.harness import build_threshold_tasks, collect_statistics


def test_repetition_circuit_has_detectors_and_observable() -> None:
    circuit = build_repetition_memory_circuit(distance=3, rounds=3, physical_error_rate=0.05)
    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_detectors > 0
    assert circuit.num_observables == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"distance": 2, "rounds": 3, "physical_error_rate": 0.05},   # even distance
        {"distance": 3, "rounds": 0, "physical_error_rate": 0.05},   # no rounds
        {"distance": 3, "rounds": 3, "physical_error_rate": 1.5},    # p out of range
    ],
)
def test_repetition_circuit_rejects_bad_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        build_repetition_memory_circuit(**kwargs)


def test_circuit_compiles_to_a_decodable_error_model() -> None:
    # A degenerate detector error model would mean PyMatching has nothing to do.
    circuit = build_repetition_memory_circuit(distance=5, rounds=5, physical_error_rate=0.03)
    dem = circuit.detector_error_model(decompose_errors=True)
    assert dem.num_detectors > 0


def test_harness_collects_a_tiny_sample() -> None:
    # End-to-end on a tiny budget: build -> sample -> get stats back.
    tasks = build_threshold_tasks(
        build_repetition_memory_circuit,
        distances=[3],
        physical_error_rates=[0.1],
        rounds=3,
    )
    assert len(tasks) == 1
    stats = collect_statistics(
        tasks, max_shots=2_000, max_errors=50, num_workers=1, print_progress=False
    )
    assert len(stats) == 1
    assert stats[0].shots > 0
