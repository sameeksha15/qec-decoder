"""Uniform Sinter-compatible wrappers for the three benchmarked decoders.

The point of this module is *fairness*: MWPM, union-find, and BP+OSD are all
exposed through the exact same ``sinter.Decoder`` / ``sinter.CompiledDecoder``
interface, with bit-packed I/O. Accuracy runs (via ``sinter.collect``) and
latency runs (via direct calls on the compiled decoders) therefore share one
identical call path per decoder, so no decoder gets a hidden advantage from
being invoked differently.

Decoder sources:

- **MWPM** — PyMatching's ``Matching``, built from the detector error model
  with error decomposition (matching needs graph-like, weight-2 errors).
- **Union-find** — ``ldpc.UnionFindDecoder`` in ``"inversion"`` (matrix-solve)
  mode — the general mode; ``"peeling"`` assumes graph-like check matrices —
  fed per-error log-likelihood ratios derived from the DEM priors.
- **BP+OSD** — ``ldpc.BpOsdDecoder`` with the DEM priors as its error channel.
  (ldpc ships its own ``SinterBpOsdDecoder``, but it only implements sinter's
  slow file-based path, not ``compile_decoder_for_dem`` — wrapping
  ``BpOsdDecoder`` ourselves keeps all three decoders on the identical
  in-memory call path, which is the fairness requirement.)

Parameter choices for BP+OSD (documented so comparisons are reproducible):
min-sum BP with the library-default scaling factor, ``max_iter = 30`` (a
standard circuit-level choice; more iterations mainly add latency), and
``osd0`` post-processing (the cheapest OSD order — the usual baseline).
"""

from __future__ import annotations

import numpy as np
import sinter
import stim
from ldpc import BpOsdDecoder, UnionFindDecoder
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
from pymatching import Matching

# BP iteration cap for BP+OSD. 30 is a common circuit-level default: BP either
# converges quickly or not at all, after which OSD takes over.
BPOSD_MAX_ITER = 30


class _CompiledMwpm(sinter.CompiledDecoder):
    """PyMatching, called through its vectorized bit-packed batch API."""

    def __init__(self, dem: stim.DetectorErrorModel) -> None:
        self._matching = Matching.from_detector_error_model(dem)
        self._num_detectors = dem.num_detectors

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        return self._matching.decode_batch(
            bit_packed_detection_event_data,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )


class SinterMwpmDecoder(sinter.Decoder):
    """Minimum-weight perfect matching (PyMatching) behind the sinter API."""

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> sinter.CompiledDecoder:
        # MWPM requires graph-like errors: decompose hyperedges into edges.
        return _CompiledMwpm(dem)


class _CompiledUnionFind(sinter.CompiledDecoder):
    """ldpc's union-find decoder driven shot-by-shot on the DEM check matrix."""

    def __init__(self, dem: stim.DetectorErrorModel) -> None:
        matrices = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True
        )
        self._check_matrix = matrices.check_matrix
        self._observables = matrices.observables_matrix
        self._num_detectors = dem.num_detectors
        self._num_observables = dem.num_observables
        # "inversion" = general matrix-solve mode; "peeling" is faster but
        # assumes a graph-like (weight-<=2) check matrix, which circuit-level
        # DEMs with hyperedges do not satisfy.
        self._decoder = UnionFindDecoder(matrices.check_matrix, uf_method="inversion")
        # Soft information: per-error log-likelihood ratios from the DEM priors.
        priors = np.clip(matrices.priors, 1e-12, 1 - 1e-12)
        self._llrs = np.log((1 - priors) / priors)

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        shots = np.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            count=self._num_detectors,
            bitorder="little",
        )
        predictions = np.zeros(
            (shots.shape[0], self._num_observables), dtype=np.uint8
        )
        for i, syndrome in enumerate(shots):
            error_estimate = self._decoder.decode(syndrome, llrs=self._llrs)
            predictions[i] = (self._observables @ error_estimate) % 2
        return np.packbits(predictions, axis=1, bitorder="little")


class SinterUnionFindDecoder(sinter.Decoder):
    """Union-find (ldpc) behind the sinter API, with prior-derived LLRs."""

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> sinter.CompiledDecoder:
        return _CompiledUnionFind(dem)


class _CompiledBpOsd(sinter.CompiledDecoder):
    """ldpc's BP+OSD decoder driven shot-by-shot on the DEM check matrix."""

    def __init__(self, dem: stim.DetectorErrorModel) -> None:
        matrices = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True
        )
        self._observables = matrices.observables_matrix
        self._num_detectors = dem.num_detectors
        self._num_observables = dem.num_observables
        self._decoder = BpOsdDecoder(
            matrices.check_matrix,
            error_channel=list(matrices.priors),
            max_iter=BPOSD_MAX_ITER,
            bp_method="ms",
            osd_method="osd0",
        )

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        shots = np.unpackbits(
            bit_packed_detection_event_data,
            axis=1,
            count=self._num_detectors,
            bitorder="little",
        )
        predictions = np.zeros(
            (shots.shape[0], self._num_observables), dtype=np.uint8
        )
        for i, syndrome in enumerate(shots):
            error_estimate = self._decoder.decode(syndrome)
            predictions[i] = (self._observables @ error_estimate) % 2
        return np.packbits(predictions, axis=1, bitorder="little")


class SinterBpOsdWrapper(sinter.Decoder):
    """BP+OSD (ldpc) behind the sinter API, priors taken from the DEM."""

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> sinter.CompiledDecoder:
        return _CompiledBpOsd(dem)


def get_custom_decoders() -> dict[str, sinter.Decoder]:
    """Registry mapping decoder names to sinter Decoder instances.

    Passed as ``custom_decoders=`` to ``sinter.collect``; the keys are the
    decoder names used in tasks, metadata, and plots.
    """
    return {
        "mwpm": SinterMwpmDecoder(),
        "unionfind": SinterUnionFindDecoder(),
        "bposd": SinterBpOsdWrapper(),
    }


DECODER_LABELS = {
    "mwpm": "MWPM (PyMatching)",
    "unionfind": "Union-Find (ldpc)",
    "bposd": f"BP+OSD-0 (ldpc, {BPOSD_MAX_ITER} iters)",
}
