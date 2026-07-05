"""Biased Pauli noise at the circuit level.

Real hardware rarely fails with symmetric depolarizing noise: many platforms
dephase far more than they bit-flip. The standard parameterization uses a total
error probability ``p`` and a bias

    eta = p_Z / (p_X + p_Y),   with p_X = p_Y,

so ``eta = 0.5`` recovers exact depolarizing noise (p_X = p_Y = p_Z = p/3) and
``eta -> infinity`` is pure dephasing. This module builds biased-noise surface
code circuits by generating Stim's standard noisy circuit and then **replacing
every depolarizing instruction with a biased Pauli channel of the same total
strength**:

- ``DEPOLARIZE1(p)``  ->  ``PAULI_CHANNEL_1(p_X, p_Y, p_Z)`` (exact at eta=0.5).
- ``DEPOLARIZE2(p)``  ->  independent ``PAULI_CHANNEL_1`` on each of the two
  qubits. Two-qubit gates do not preserve bias in general, and modelling their
  noise as independent biased single-qubit channels is the common circuit-level
  convention; note this is *not* identical to ``DEPOLARIZE2`` even at
  eta = 0.5 (a product of single-qubit channels cannot reproduce the uniform
  two-qubit Pauli distribution), which is why experiments refit their
  eta = 0.5 baseline instead of comparing against the DEPOLARIZE2 numbers.
- ``X_ERROR`` (measurement/reset flips) is left untouched: SPAM errors are
  classical flips with no Pauli structure to bias.

The transformation recurses into REPEAT blocks, so the generated circuit's
round structure survives intact.
"""

from __future__ import annotations

import stim


def biased_pauli_probabilities(
    physical_error_rate: float, bias_eta: float
) -> tuple[float, float, float]:
    """Split a total error rate into (p_X, p_Y, p_Z) at the given bias.

    Args:
        physical_error_rate: Total probability ``p`` of any Pauli error,
            in ``[0, 1]``.
        bias_eta: Bias ``eta = p_Z / (p_X + p_Y)`` with ``p_X = p_Y``. Must be
            > 0; ``0.5`` is depolarizing, large values are dephasing-dominated.

    Returns:
        ``(p_X, p_Y, p_Z)`` summing to ``physical_error_rate``.
    """
    if not 0.0 <= physical_error_rate <= 1.0:
        raise ValueError(
            f"physical_error_rate must be in [0, 1], got {physical_error_rate}"
        )
    if bias_eta <= 0:
        raise ValueError(f"bias_eta must be > 0, got {bias_eta}")

    p = physical_error_rate
    p_z = p * bias_eta / (1.0 + bias_eta)
    p_x = p_y = p / (2.0 * (1.0 + bias_eta))
    return p_x, p_y, p_z


def apply_bias_to_depolarizing_circuit(
    circuit: stim.Circuit, *, bias_eta: float
) -> stim.Circuit:
    """Replace every depolarizing instruction with a biased Pauli channel.

    Each ``DEPOLARIZE1(p)`` / ``DEPOLARIZE2(p)`` becomes single-qubit
    ``PAULI_CHANNEL_1(p_X, p_Y, p_Z)`` noise of the same total strength ``p``
    on the same qubits (independently per qubit for the two-qubit case).
    All other instructions — gates, measurements, resets, detectors,
    observables, X_ERROR flips — pass through unchanged. REPEAT blocks are
    transformed recursively.

    Args:
        circuit: A noisy circuit whose noise is expressed via DEPOLARIZE1/2
            (e.g. from ``stim.Circuit.generated``).
        bias_eta: The bias to apply; 0.5 leaves single-qubit channels exactly
            depolarizing.

    Returns:
        A new ``stim.Circuit`` with identical structure and biased noise.
    """
    result = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(
                stim.CircuitRepeatBlock(
                    instruction.repeat_count,
                    apply_bias_to_depolarizing_circuit(
                        instruction.body_copy(), bias_eta=bias_eta
                    ),
                )
            )
        elif instruction.name in ("DEPOLARIZE1", "DEPOLARIZE2"):
            (strength,) = instruction.gate_args_copy()
            result.append(
                "PAULI_CHANNEL_1",
                instruction.targets_copy(),
                biased_pauli_probabilities(strength, bias_eta),
            )
        else:
            result.append(instruction)
    return result


def build_biased_surface_code_memory_circuit(
    *,
    distance: int,
    rounds: int,
    physical_error_rate: float,
    bias_eta: float,
    basis: str = "x",
) -> stim.Circuit:
    """Build a rotated-surface-code memory experiment under biased noise.

    Args:
        distance: Code distance ``d`` (odd, >= 3).
        rounds: Stabilizer measurement rounds (>= 1).
        physical_error_rate: Total Pauli error probability per noise site.
        bias_eta: Noise bias ``eta = p_Z / (p_X + p_Y)``; 0.5 is depolarizing.
        basis: ``"x"`` or ``"z"`` memory. Under Z-biased noise (eta > 0.5) the
            X-basis memory is the *stressed* direction — its logical observable
            fails via Z errors — so benchmarks default to ``"x"``. The Z-basis
            memory is the mirror case and mainly useful for validation.

    Returns:
        A ``stim.Circuit`` with detectors and one logical observable.

    Raises:
        ValueError: If any argument is outside its valid range.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be an odd integer >= 3, got {distance}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if basis not in ("x", "z"):
        raise ValueError(f"basis must be 'x' or 'z', got {basis!r}")
    # physical_error_rate and bias_eta are validated in
    # biased_pauli_probabilities via the transformation below.

    depolarizing = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=physical_error_rate,
        before_round_data_depolarization=physical_error_rate,
        before_measure_flip_probability=physical_error_rate,
        after_reset_flip_probability=physical_error_rate,
    )
    return apply_bias_to_depolarizing_circuit(depolarizing, bias_eta=bias_eta)
