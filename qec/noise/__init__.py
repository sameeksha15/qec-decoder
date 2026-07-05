"""Circuit-level noise models.

Stim's circuit generators bake in uniform depolarizing noise; the models here
transform such circuits into structured-noise variants (applied to explicit
instructions rather than generator knobs).
"""

from qec.noise.biased import (
    apply_bias_to_depolarizing_circuit,
    biased_pauli_probabilities,
    build_biased_surface_code_memory_circuit,
)

__all__ = [
    "apply_bias_to_depolarizing_circuit",
    "biased_pauli_probabilities",
    "build_biased_surface_code_memory_circuit",
]
