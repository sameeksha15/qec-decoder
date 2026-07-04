"""Decoder-layer tests: uniform decoder wrappers and the latency harness.

Fast, single-process checks: every decoder in the registry can compile from a
DEM, decode a real syndrome batch through the shared bit-packed interface, and
produce sane logical error counts. No multiprocessing involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.codes import build_surface_code_memory_circuit
from qec.decoders import DECODER_LABELS, get_custom_decoders
from qec.harness import benchmark_decoder_latency

DISTANCE = 3
SHOTS = 500
P = 0.008


@pytest.fixture(scope="module")
def circuit():
    return build_surface_code_memory_circuit(
        distance=DISTANCE, rounds=DISTANCE, physical_error_rate=P
    )


@pytest.fixture(scope="module")
def sampled(circuit):
    sampler = circuit.compile_detector_sampler(seed=7)
    dets, obs = sampler.sample(
        shots=SHOTS, separate_observables=True, bit_packed=True
    )
    return dets, obs


def test_registry_and_labels_are_consistent() -> None:
    decoders = get_custom_decoders()
    assert set(decoders) == {"mwpm", "unionfind", "bposd"}
    assert set(DECODER_LABELS) == set(decoders)


@pytest.mark.parametrize("name", ["mwpm", "unionfind", "bposd"])
def test_decoder_decodes_real_shots_with_sane_accuracy(name, circuit, sampled) -> None:
    dets, actual_obs = sampled
    decoder = get_custom_decoders()[name]
    dem = circuit.detector_error_model(decompose_errors=True)
    compiled = decoder.compile_decoder_for_dem(dem=dem)

    predictions = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=dets
    )
    assert predictions.shape == actual_obs.shape

    # Sanity band, not a benchmark: at p = 0.008 and d = 3 a working decoder
    # should mispredict a few percent of shots; a broken one sits near 50%.
    error_rate = np.mean(
        np.unpackbits(predictions ^ actual_obs, bitorder="little")
    ) * 8  # one observable byte per shot -> x8 undoes the bit-dilution
    assert error_rate < 0.25, f"{name} logical error rate {error_rate:.3f}"


def test_latency_benchmark_returns_positive_times(circuit) -> None:
    results = benchmark_decoder_latency(
        circuit,
        get_custom_decoders(),
        num_shots=100,
        warmup_shots=10,
        repeats=2,
    )
    assert {r.decoder_name for r in results} == {"mwpm", "unionfind", "bposd"}
    assert all(r.seconds_per_shot > 0 for r in results)


def test_latency_benchmark_rejects_bad_arguments(circuit) -> None:
    with pytest.raises(ValueError):
        benchmark_decoder_latency(
            circuit, get_custom_decoders(), num_shots=0
        )
