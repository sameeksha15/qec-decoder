"""Code-capacity memory experiments for arbitrary CSS codes.

The circuit-level experiments elsewhere in this package rely on Stim's
generators, which only exist for standard codes. To study *any* CSS code
(e.g. bivariate-bicycle codes) on the same Stim -> Sinter -> decoder pipeline,
this module builds **code-capacity** memory circuits directly from the check
matrices: stabilizers are measured perfectly via ``MPP`` (multi-Pauli product)
instructions, and noise is a single layer of iid depolarizing noise on the
data qubits.

Code capacity is a deliberately idealized model — no faulty syndrome
extraction — so numbers from it are *not* comparable to circuit-level results.
Its value is comparative: two codes measured under the identical idealized
model, with the identical decoder, reveal their intrinsic error-correcting
power per qubit.

Circuit shape (Z-basis memory):

    R  all data                          | prepare |0...0>
    MPP all stabilizers  (reference)     | noiseless round 1
    DEPOLARIZE1(p) all data              | the only noise in the model
    MPP all stabilizers  (syndrome)      | noiseless round 2
    DETECTOR (round 1 vs round 2)        | one per stabilizer
    DETECTOR (round 1 alone)             | Z-stabilizers are deterministic
                                         |   after R, so round 1 itself checks
    MPP each logical Z -> OBSERVABLE     | k observables; any flip = failure
"""

from __future__ import annotations

import numpy as np
import stim

from qec.codes.bivariate_bicycle import CssCode


def _mpp_pauli_product(circuit: stim.Circuit, pauli: str, support: np.ndarray) -> None:
    """Append one MPP measurement of a Pauli product to the circuit."""
    target = {"X": stim.target_x, "Z": stim.target_z}[pauli]
    qubits = np.nonzero(support)[0]
    targets: list[stim.GateTarget] = []
    for i, qubit in enumerate(qubits):
        if i:
            targets.append(stim.target_combiner())
        targets.append(target(int(qubit)))
    circuit.append("MPP", targets)


def build_css_code_capacity_circuit(
    code: CssCode,
    *,
    physical_error_rate: float,
) -> stim.Circuit:
    """Build a Z-basis code-capacity memory circuit for a CSS code.

    Args:
        code: The CSS code (check matrices + verified logical operators).
        physical_error_rate: Depolarizing probability applied once to every
            data qubit between two perfect syndrome-extraction rounds.

    Returns:
        A ``stim.Circuit`` with one detector pair per stabilizer and one
        observable per logical qubit (a shot fails if *any* logical flips —
        the per-block failure rate).

    Raises:
        ValueError: If the error rate is outside [0, 1].
    """
    if not 0.0 <= physical_error_rate <= 1.0:
        raise ValueError(
            f"physical_error_rate must be in [0, 1], got {physical_error_rate}"
        )

    n = code.num_qubits
    stabilizers = [("X", row) for row in code.hx] + [("Z", row) for row in code.hz]
    num_stabilizers = len(stabilizers)

    circuit = stim.Circuit()
    circuit.append("R", range(n))

    for pauli, row in stabilizers:            # round 1: reference
        _mpp_pauli_product(circuit, pauli, row)
    circuit.append("DEPOLARIZE1", range(n), physical_error_rate)
    for pauli, row in stabilizers:            # round 2: syndrome
        _mpp_pauli_product(circuit, pauli, row)

    # Compare the two measurement rounds of each stabilizer.
    for i in range(num_stabilizers):
        circuit.append(
            "DETECTOR",
            [
                stim.target_rec(i - 2 * num_stabilizers),
                stim.target_rec(i - num_stabilizers),
            ],
        )
    # Z-stabilizers are deterministic (+1) on |0...0>, so their reference
    # round is itself a parity check on state preparation.
    num_x = code.hx.shape[0]
    for i in range(num_x, num_stabilizers):
        circuit.append("DETECTOR", [stim.target_rec(i - 2 * num_stabilizers)])

    # Measure every logical Z; each is its own observable.
    for index, logical in enumerate(code.logical_z):
        _mpp_pauli_product(circuit, "Z", logical)
        circuit.append(
            "OBSERVABLE_INCLUDE", [stim.target_rec(-1)], index
        )
    return circuit
