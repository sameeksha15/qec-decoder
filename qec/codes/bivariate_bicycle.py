"""Bivariate-bicycle (BB) quantum LDPC codes.

BB codes (Bravyi et al., Nature 627, 778 (2024)) are CSS codes built from two
polynomials over a pair of commuting cyclic shifts. With ``x`` the shift on a
ring of size ``l`` and ``y`` the shift on a ring of size ``m``:

    A = sum of monomials  x^a y^b     (three terms in the standard family)
    B = sum of monomials  x^c y^d

    H_X = [A | B]           (m*l rows, 2*m*l columns)
    H_Z = [B^T | A^T]

CSS commutation H_X @ H_Z^T = A B + B A = 0 holds because A and B are both
polynomials in the same commuting shifts. The family broke the assumption that
the surface code is the only practical option: comparable protection at a
much higher encoding rate (the [[144, 12, 12]] "gross code" needs ~10x fewer
qubits per logical qubit than surface patches of similar distance).

This module constructs the check matrices from the polynomial exponents and
*verifies* the code parameters with explicit GF(2) linear algebra (rank,
kernel, logical-operator extraction) rather than trusting the literature
numbers. The distance is the one parameter we cite rather than compute —
exact distance calculation is NP-hard in general and out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The smallest standard code of the family (Bravyi et al., Table 3):
# l = m = 6, A = x^3 + y + y^2, B = y^3 + x + x^2  ->  [[72, 12, 6]].
BB_72_12_6 = {
    "l": 6,
    "m": 6,
    "a_terms": ((3, 0), (0, 1), (0, 2)),
    "b_terms": ((0, 3), (1, 0), (2, 0)),
    "cited_distance": 6,
}


# --- GF(2) linear algebra ----------------------------------------------------


def gf2_row_reduce(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Row-reduce a binary matrix over GF(2).

    Returns:
        ``(rref, pivot_columns)`` where ``rref`` is the reduced matrix
        (dtype uint8) and ``pivot_columns`` lists the pivot column of each
        nonzero row.
    """
    m = np.array(matrix, dtype=np.uint8, copy=True) % 2
    pivots: list[int] = []
    row = 0
    for col in range(m.shape[1]):
        if row >= m.shape[0]:
            break
        support = np.nonzero(m[row:, col])[0]
        if support.size == 0:
            continue
        m[[row, row + support[0]]] = m[[row + support[0], row]]
        eliminate = np.nonzero(m[:, col])[0]
        eliminate = eliminate[eliminate != row]
        m[eliminate] ^= m[row]
        pivots.append(col)
        row += 1
    return m, pivots


def gf2_rank(matrix: np.ndarray) -> int:
    """Rank of a binary matrix over GF(2)."""
    _, pivots = gf2_row_reduce(matrix)
    return len(pivots)


def gf2_kernel_basis(matrix: np.ndarray) -> np.ndarray:
    """Basis of the right kernel {v : M v = 0 (mod 2)} as rows."""
    rref, pivots = gf2_row_reduce(matrix)
    n = rref.shape[1]
    free_columns = [c for c in range(n) if c not in pivots]
    basis = np.zeros((len(free_columns), n), dtype=np.uint8)
    for i, free in enumerate(free_columns):
        basis[i, free] = 1
        for row_index, pivot in enumerate(pivots):
            if rref[row_index, free]:
                basis[i, pivot] = 1
    return basis


def _quotient_basis(candidates: np.ndarray, subspace: np.ndarray, count: int) -> np.ndarray:
    """Pick ``count`` candidate rows independent of ``subspace`` (mod 2).

    Greedy selection: keep a candidate iff it increases the rank of the
    stack [subspace; kept so far]. Used to extract logical operators as a
    basis of kernel / stabilizer-rowspace.
    """
    kept: list[np.ndarray] = []
    stack = subspace.copy()
    rank = gf2_rank(stack)
    for row in candidates:
        trial = np.vstack([stack, row[None, :]])
        trial_rank = gf2_rank(trial)
        if trial_rank > rank:
            kept.append(row)
            stack, rank = trial, trial_rank
            if len(kept) == count:
                break
    if len(kept) != count:
        raise ValueError(
            f"could not extract {count} independent representatives "
            f"(found {len(kept)})"
        )
    return np.array(kept, dtype=np.uint8)


# --- BB code construction -----------------------------------------------------


@dataclass(frozen=True)
class CssCode:
    """A CSS code as explicit GF(2) data, with verified parameters."""

    name: str
    hx: np.ndarray            # X-type stabilizer checks (detect Z errors)
    hz: np.ndarray            # Z-type stabilizer checks (detect X errors)
    logical_x: np.ndarray     # k logical-X representatives (rows)
    logical_z: np.ndarray     # k logical-Z representatives (rows)
    cited_distance: int | None = None

    @property
    def num_qubits(self) -> int:
        return self.hx.shape[1]

    @property
    def num_logicals(self) -> int:
        return self.logical_z.shape[0]

    @property
    def encoding_rate(self) -> float:
        return self.num_logicals / self.num_qubits


def _cyclic_shift(size: int, power: int) -> np.ndarray:
    return np.roll(np.eye(size, dtype=np.uint8), power % size, axis=1)


def _polynomial_matrix(l: int, m: int, terms) -> np.ndarray:
    """Sum (mod 2) of monomials x^a y^b as an lm x lm binary matrix."""
    total = np.zeros((l * m, l * m), dtype=np.uint8)
    for a, b in terms:
        total ^= np.kron(_cyclic_shift(l, a), _cyclic_shift(m, b))
    return total


def assemble_css_code(
    name: str,
    hx: np.ndarray,
    hz: np.ndarray,
    *,
    cited_distance: int | None = None,
) -> CssCode:
    """Verify CSS check matrices and extract logical operators.

    Shared by every explicit-matrix code in this package: checks commutation,
    computes ``k = n - rank(H_X) - rank(H_Z)``, and derives logical operator
    representatives via kernel/rowspace quotients.

    Raises:
        ValueError: If the checks do not commute or the code encodes nothing.
    """
    hx = np.asarray(hx, dtype=np.uint8) % 2
    hz = np.asarray(hz, dtype=np.uint8) % 2
    if hx.shape[1] != hz.shape[1]:
        raise ValueError("H_X and H_Z must act on the same number of qubits")
    if np.any((hx @ hz.T) % 2):
        raise ValueError("H_X and H_Z do not commute; invalid construction")

    n = hx.shape[1]
    k = n - gf2_rank(hx) - gf2_rank(hz)
    if k <= 0:
        raise ValueError(f"construction encodes no logical qubits (k={k})")

    # Logical Z: commutes with X-checks (kernel of H_X), not itself a Z-check.
    logical_z = _quotient_basis(gf2_kernel_basis(hx), hz, k)
    # Logical X: commutes with Z-checks, not itself an X-check.
    logical_x = _quotient_basis(gf2_kernel_basis(hz), hx, k)

    return CssCode(
        name=name,
        hx=hx,
        hz=hz,
        logical_x=logical_x,
        logical_z=logical_z,
        cited_distance=cited_distance,
    )


def build_bivariate_bicycle_code(
    *,
    l: int,
    m: int,
    a_terms,
    b_terms,
    cited_distance: int | None = None,
    name: str | None = None,
) -> CssCode:
    """Construct and verify a bivariate-bicycle code.

    Args:
        l: Size of the x-shift ring (must be >= 1).
        m: Size of the y-shift ring (must be >= 1).
        a_terms: Iterable of (x-exponent, y-exponent) monomials of A.
        b_terms: Iterable of (x-exponent, y-exponent) monomials of B.
        cited_distance: Literature distance, carried as metadata (not
            computed — exact distance is NP-hard in general).
        name: Optional label; defaults to the standard [[n, k, d]] form.

    Returns:
        A :class:`CssCode` whose stabilizer commutation and logical operators
        have been explicitly verified over GF(2).

    Raises:
        ValueError: If the construction fails verification (non-commuting
            checks, or k = 0).
    """
    if l < 1 or m < 1:
        raise ValueError(f"ring sizes must be >= 1, got l={l}, m={m}")

    a = _polynomial_matrix(l, m, a_terms)
    b = _polynomial_matrix(l, m, b_terms)
    hx = np.concatenate([a, b], axis=1)
    hz = np.concatenate([b.T, a.T], axis=1)

    n = hx.shape[1]
    label = name or f"BB [[{n}, ?, {cited_distance or '?'}]]"
    code = assemble_css_code(label, hx, hz, cited_distance=cited_distance)
    if name is None:
        code = CssCode(
            name=f"BB [[{n}, {code.num_logicals}, {cited_distance or '?'}]]",
            hx=code.hx,
            hz=code.hz,
            logical_x=code.logical_x,
            logical_z=code.logical_z,
            cited_distance=cited_distance,
        )
    return code


def bb_72_12_6() -> CssCode:
    """The [[72, 12, 6]] bivariate-bicycle code (smallest standard instance)."""
    return build_bivariate_bicycle_code(**BB_72_12_6)
