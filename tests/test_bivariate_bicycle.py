"""Bivariate-bicycle code tests: construction, GF(2) algebra, code capacity.

The headline check: the [[72, 12, 6]] construction must *prove* its n and k by
explicit rank computation, its stabilizers must commute, and its logical
operators must satisfy the defining (anti)commutation relations — nothing is
taken on faith from the literature except the distance.
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.codes import bb_72_12_6, build_css_code_capacity_circuit
from qec.codes.bivariate_bicycle import (
    build_bivariate_bicycle_code,
    gf2_kernel_basis,
    gf2_rank,
)


@pytest.fixture(scope="module")
def code():
    return bb_72_12_6()


def test_gf2_rank_and_kernel_basics() -> None:
    m = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    assert gf2_rank(m) == 2
    kernel = gf2_kernel_basis(m)
    assert kernel.shape == (1, 3)
    assert not np.any((m @ kernel.T) % 2)


def test_bb_72_12_6_parameters(code) -> None:
    assert code.num_qubits == 72
    assert code.num_logicals == 12
    assert code.encoding_rate == pytest.approx(12 / 72)
    assert code.cited_distance == 6


def test_stabilizers_commute(code) -> None:
    assert not np.any((code.hx @ code.hz.T) % 2)


def test_logical_operators_satisfy_css_relations(code) -> None:
    # Logical Z commutes with every X-check; logical X with every Z-check.
    assert not np.any((code.hx @ code.logical_z.T) % 2)
    assert not np.any((code.hz @ code.logical_x.T) % 2)
    # Logicals are independent of the stabilizer group (k of them survive
    # the quotient), which build_bivariate_bicycle_code enforces; check rank.
    assert gf2_rank(np.vstack([code.hz, code.logical_z])) == gf2_rank(code.hz) + 12
    assert gf2_rank(np.vstack([code.hx, code.logical_x])) == gf2_rank(code.hx) + 12
    # The X and Z logical spaces pair nontrivially (full-rank overlap matrix),
    # i.e. they really address the same 12 logical qubits.
    assert gf2_rank((code.logical_x @ code.logical_z.T) % 2) == 12


def test_invalid_construction_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_bivariate_bicycle_code(l=0, m=6, a_terms=[(0, 0)], b_terms=[(0, 0)])


def test_code_capacity_circuit_is_well_formed(code) -> None:
    circuit = build_css_code_capacity_circuit(code, physical_error_rate=0.05)
    # One comparison detector per stabilizer + one deterministic detector per
    # Z-stabilizer reference measurement.
    num_stabilizers = code.hx.shape[0] + code.hz.shape[0]
    assert circuit.num_detectors == num_stabilizers + code.hz.shape[0]
    assert circuit.num_observables == 12
    # Must compile to a decodable error model without error decomposition
    # (BB hyperedges are not graph-like; BP+OSD consumes them natively).
    dem = circuit.detector_error_model(decompose_errors=False)
    assert dem.num_detectors == circuit.num_detectors


@pytest.mark.parametrize("distance", [3, 5])
def test_rotated_surface_css_construction(distance) -> None:
    from qec.codes import rotated_surface_css_code

    code = rotated_surface_css_code(distance)
    assert code.num_qubits == distance**2
    assert code.num_logicals == 1
    assert code.hx.shape[0] == (distance**2 - 1) // 2
    assert code.hz.shape[0] == (distance**2 - 1) // 2
    # The rotated layout's logicals are a single row/column: weight exactly d.
    assert code.logical_x.sum() == distance
    assert code.logical_z.sum() == distance


def test_rotated_surface_css_rejects_even_distance() -> None:
    from qec.codes import rotated_surface_css_code

    with pytest.raises(ValueError):
        rotated_surface_css_code(4)


def test_code_capacity_end_to_end_with_bposd(code) -> None:
    # Below-threshold sanity: at p = 0.01, BP+OSD on the [[72,12,6]] block
    # should fail only a few percent of shots; a broken pipeline sits near 50%.
    from qec.decoders import get_custom_decoders

    circuit = build_css_code_capacity_circuit(code, physical_error_rate=0.01)
    dem = circuit.detector_error_model(decompose_errors=False)
    compiled = get_custom_decoders()["bposd"].compile_decoder_for_dem(dem=dem)

    dets, obs = circuit.compile_detector_sampler(seed=5).sample(
        shots=300, separate_observables=True, bit_packed=True
    )
    predictions = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=dets
    )
    block_failures = np.mean(np.any(predictions != obs, axis=1))
    assert block_failures < 0.25, f"block failure rate {block_failures:.3f}"
