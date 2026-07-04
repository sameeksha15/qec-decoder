"""The rotated surface code memory experiment.

The surface code is the workhorse of fault-tolerant quantum computing: a 2-D grid
of data qubits with two interleaved families of stabilizers (X-type and Z-type)
that detect phase-flip and bit-flip errors respectively. Unlike the repetition
code, it is a genuine quantum code — it corrects *both* error types —
and its planar, nearest-neighbour layout matches real superconducting and
neutral-atom hardware.

The "rotated" variant uses the minimal d x d data-qubit patch for a given code
distance ``d``, which is the standard experimental layout.

As with the repetition code we use Stim's built-in generator. It emits a circuit-level noisy
memory experiment whose detectors compare consecutive rounds of stabilizer
measurements, with a single logical observable measured at the end. The function
below wraps it behind the same ``code_builder`` contract the harness expects, so
the entire sampling/threshold machinery is reused unchanged.
"""

from __future__ import annotations

import stim

# We study the Z-basis memory ("memory_z"): prepare logical |0>, idle for several
# rounds of noisy stabilizer measurement, then measure. A logical bit-flip over
# the experiment is the failure we count. (memory_x is the mirror image; one
# basis suffices to locate the threshold.)
_GENERATOR_NAME = "surface_code:rotated_memory_z"


def build_surface_code_memory_circuit(
    *,
    distance: int,
    rounds: int,
    physical_error_rate: float,
) -> stim.Circuit:
    """Build a noisy rotated-surface-code memory experiment.

    Args:
        distance: Code distance ``d``. Must be an odd integer >= 3 (the rotated
            patch uses a d x d grid of data qubits; odd ``d`` gives the standard
            single-logical-qubit patch).
        rounds: Number of stabilizer measurement rounds. Must be >= 1; the usual
            convention is ``rounds = d`` so the experiment depth scales with the
            code.
        physical_error_rate: Single probability ``p`` in ``[0, 1]`` driving every
            circuit-level noise channel (gate, data, measurement, reset).

    Returns:
        A ``stim.Circuit`` carrying X- and Z-type detectors and one logical
        observable, ready to be compiled into a detector error model and sampled.

    Raises:
        ValueError: If any argument is outside its valid range.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be an odd integer >= 3, got {distance}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if not 0.0 <= physical_error_rate <= 1.0:
        raise ValueError(
            f"physical_error_rate must be in [0, 1], got {physical_error_rate}"
        )

    return stim.Circuit.generated(
        _GENERATOR_NAME,
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=physical_error_rate,
        before_round_data_depolarization=physical_error_rate,
        before_measure_flip_probability=physical_error_rate,
        after_reset_flip_probability=physical_error_rate,
    )
