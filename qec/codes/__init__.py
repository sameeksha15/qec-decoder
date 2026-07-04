"""Code and circuit constructors.

The repetition code (a classical-style bit-flip code) exists purely as a
harness shake-down target; every constructor honours the same
"return a ``stim.Circuit``" contract.
"""

from qec.codes.repetition import build_repetition_memory_circuit
from qec.codes.surface import build_surface_code_memory_circuit

__all__ = [
    "build_repetition_memory_circuit",
    "build_surface_code_memory_circuit",
]
