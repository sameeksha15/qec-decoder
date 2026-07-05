"""The rotated surface code as explicit CSS check matrices.

The circuit-level experiments use Stim's generator; this module provides the
same code as raw ``(H_X, H_Z)`` matrices so it can run through the generic
code-capacity pipeline (``build_css_code_capacity_circuit``) on an equal
footing with matrix-defined codes like the bivariate-bicycle family.

Layout: data qubits on a d x d grid. Stabilizers live on the (d+1) x (d+1)
grid of plaquette corners; the plaquette at (r, c) touches the up-to-four data
qubits {(r-1, c-1), (r-1, c), (r, c-1), (r, c)} that exist. Bulk plaquettes
(weight 4) alternate X/Z in a checkerboard; boundary plaquettes (weight 2)
are kept only where their type's boundary runs (X on top/bottom, Z on
left/right), which is what pins k = 1. The construction is *verified* — rank,
commutation, logical operators — by ``assemble_css_code``, not trusted.
"""

from __future__ import annotations

import numpy as np

from qec.codes.bivariate_bicycle import CssCode, assemble_css_code


def rotated_surface_css_code(distance: int) -> CssCode:
    """Build the rotated surface code [[d^2, 1, d]] as a :class:`CssCode`.

    Args:
        distance: Code distance ``d`` (odd, >= 3).

    Returns:
        The verified code, with ``cited_distance = d`` (for the rotated
        layout the row/column logicals make the distance manifest).

    Raises:
        ValueError: If ``distance`` is invalid or verification fails.
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError(f"distance must be an odd integer >= 3, got {distance}")

    d = distance
    index = {(row, col): row * d + col for row in range(d) for col in range(d)}

    x_checks: list[np.ndarray] = []
    z_checks: list[np.ndarray] = []
    for r in range(d + 1):
        for c in range(d + 1):
            support = np.zeros(d * d, dtype=np.uint8)
            for row, col in ((r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c)):
                if 0 <= row < d and 0 <= col < d:
                    support[index[row, col]] = 1
            weight = int(support.sum())
            if weight < 2:
                continue
            is_x = (r + c) % 2 == 0
            if weight == 4:
                (x_checks if is_x else z_checks).append(support)
            elif weight == 2:
                # Boundary checks survive only on their own boundary: X-type
                # on the top/bottom rows, Z-type on the left/right columns.
                if is_x and r in (0, d):
                    x_checks.append(support)
                elif not is_x and c in (0, d):
                    z_checks.append(support)

    return assemble_css_code(
        f"rotated surface [[{d * d}, 1, {d}]]",
        np.array(x_checks, dtype=np.uint8),
        np.array(z_checks, dtype=np.uint8),
        cited_distance=d,
    )
