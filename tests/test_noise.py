"""Biased-noise model tests.

The load-bearing check: at eta = 0.5 a transformed single-qubit-noise circuit
must produce a detector error model *identical* to the depolarizing original —
that equivalence is what anchors the whole bias sweep to the validated
depolarizing baseline.
"""

from __future__ import annotations

import pytest
import stim

from qec.noise import (
    apply_bias_to_depolarizing_circuit,
    biased_pauli_probabilities,
    build_biased_surface_code_memory_circuit,
)


def test_probabilities_sum_to_total_and_respect_bias() -> None:
    p_x, p_y, p_z = biased_pauli_probabilities(0.01, bias_eta=10.0)
    assert p_x == pytest.approx(p_y)
    assert p_x + p_y + p_z == pytest.approx(0.01)
    assert p_z / (p_x + p_y) == pytest.approx(10.0)


def test_eta_half_is_exactly_depolarizing() -> None:
    p_x, p_y, p_z = biased_pauli_probabilities(0.03, bias_eta=0.5)
    assert p_x == pytest.approx(0.01)
    assert p_y == pytest.approx(0.01)
    assert p_z == pytest.approx(0.01)


@pytest.mark.parametrize("bad", [{"physical_error_rate": -0.1}, {"bias_eta": 0.0}, {"bias_eta": -3}])
def test_rejects_bad_arguments(bad) -> None:
    kwargs = {"physical_error_rate": 0.01, "bias_eta": 1.0, **bad}
    with pytest.raises(ValueError):
        biased_pauli_probabilities(kwargs["physical_error_rate"], kwargs["bias_eta"])


def _canonical_error_set(dem: stim.DetectorErrorModel) -> set:
    """A DEM as a set of (probability, decomposition components).

    Stim orders decomposition components (``A ^ B`` vs ``B ^ A``) differently
    for PAULI_CHANNEL_1 than for DEPOLARIZE1, so canonical text equality
    (``approx_equals``) fails even when the models are physically identical.
    Comparing unordered component sets is the physically meaningful equality.
    """
    errors = set()
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        (probability,) = instruction.args_copy()
        components = frozenset(
            tuple(sorted(str(t) for t in group))
            for group in instruction.target_groups()
        )
        errors.add((round(probability, 10), components))
    return errors


def test_eta_half_dem_matches_depolarizing_for_single_qubit_noise() -> None:
    # Only single-qubit depolarizing noise (two-qubit gate noise off), so the
    # eta = 0.5 transformation must be an exact identity at the DEM level.
    depolarizing = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=3,
        distance=3,
        before_round_data_depolarization=0.01,
    )
    biased = apply_bias_to_depolarizing_circuit(depolarizing, bias_eta=0.5)
    assert _canonical_error_set(
        biased.detector_error_model(decompose_errors=True)
    ) == _canonical_error_set(
        depolarizing.detector_error_model(decompose_errors=True)
    )


def test_transformation_recurses_into_repeat_blocks() -> None:
    circuit = build_biased_surface_code_memory_circuit(
        distance=3, rounds=5, physical_error_rate=0.01, bias_eta=10.0
    )
    text = str(circuit)
    assert "DEPOLARIZE" not in text, "some depolarizing noise was not transformed"
    assert "REPEAT" in text, "round structure (REPEAT block) was lost"
    assert "PAULI_CHANNEL_1" in text


def test_biased_circuit_structure_matches_original() -> None:
    biased = build_biased_surface_code_memory_circuit(
        distance=3, rounds=3, physical_error_rate=0.008, bias_eta=30.0
    )
    assert biased.num_detectors > 0
    assert biased.num_observables == 1
    # Must remain decodable end to end.
    dem = biased.detector_error_model(decompose_errors=True)
    assert dem.num_detectors == biased.num_detectors


def test_strong_z_bias_stresses_x_memory_more_than_z_memory() -> None:
    # Physics sanity check: under heavy Z bias the X-basis memory (whose
    # logical observable fails via Z errors) must be strictly worse than the
    # Z-basis memory on identical sampling budgets.
    import numpy as np
    import pymatching

    rates = {}
    for basis in ("x", "z"):
        circuit = build_biased_surface_code_memory_circuit(
            distance=3, rounds=3, physical_error_rate=0.01, bias_eta=100.0,
            basis=basis,
        )
        dets, obs = circuit.compile_detector_sampler(seed=11).sample(
            shots=4_000, separate_observables=True
        )
        matching = pymatching.Matching.from_detector_error_model(
            circuit.detector_error_model(decompose_errors=True)
        )
        predictions = matching.decode_batch(dets)
        rates[basis] = np.mean(np.any(predictions != obs, axis=1))

    assert rates["x"] > 2 * rates["z"], rates
