"""The repetition code memory experiment.

The repetition code is the "hello world" of quantum error correction. It only
protects against a single error type (bit flips), so it is *not* a real quantum
code — but its detector error model is a simple 1D matching graph, which makes
it the ideal target for validating that our Stim -> Sinter -> PyMatching
pipeline is wired up correctly before we trust it on the surface code.

We lean on Stim's built-in circuit generator here rather than laying out gates by
hand. For a standard code that is the idiomatic choice; less-standard circuits
are constructed explicitly elsewhere in this package.
"""

from __future__ import annotations

import stim

# Stim's generated repetition circuit exposes several independent noise knobs.
# For a first phenomenological study we drive them all from a single physical
# error rate ``p`` so the threshold plot has one clean x-axis.
_GENERATOR_NAME = "repetition_code:memory"


def build_repetition_memory_circuit(
    *,
    distance: int,
    rounds: int,
    physical_error_rate: float,
) -> stim.Circuit:
    """Build a noisy repetition-code memory experiment.

    Args:
        distance: Number of data qubits / code distance ``d``. Must be odd and
            >= 3 so that majority-vote decoding is well defined.
        rounds: Number of stabilizer measurement rounds. Must be >= 1.
        physical_error_rate: Single probability ``p`` in ``[0, 1]`` applied to
            data depolarization, measurement flips, and reset/gate noise.

    Returns:
        A ``stim.Circuit`` carrying detectors and a logical observable, ready to
        be compiled into a detector error model and sampled.

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
        before_round_data_depolarization=physical_error_rate,
        before_measure_flip_probability=physical_error_rate,
        after_clifford_depolarization=physical_error_rate,
        after_reset_flip_probability=physical_error_rate,
    )
