"""Code and circuit constructors.

The repetition code (a classical-style bit-flip code) exists purely as a
harness shake-down target; every constructor honours the same
"return a ``stim.Circuit``" contract.
"""

from qec.codes.bivariate_bicycle import (
    CssCode,
    assemble_css_code,
    bb_72_12_6,
    build_bivariate_bicycle_code,
)
from qec.codes.css_capacity import build_css_code_capacity_circuit
from qec.codes.surface_css import rotated_surface_css_code
from qec.codes.repetition import build_repetition_memory_circuit
from qec.codes.surface import build_surface_code_memory_circuit

__all__ = [
    "CssCode",
    "assemble_css_code",
    "bb_72_12_6",
    "build_bivariate_bicycle_code",
    "build_css_code_capacity_circuit",
    "build_repetition_memory_circuit",
    "build_surface_code_memory_circuit",
    "rotated_surface_css_code",
]
