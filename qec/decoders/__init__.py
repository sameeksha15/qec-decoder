"""Uniform decoder interface for the decoder benchmarks.

Every decoder (MWPM, union-find, BP+OSD) is exposed through the same
``sinter.Decoder`` contract so accuracy and latency measurements share one
identical call path. See :mod:`qec.decoders.wrappers` for the rationale and
the documented parameter choices.
"""

from qec.decoders.wrappers import (
    DECODER_LABELS,
    SinterMwpmDecoder,
    SinterUnionFindDecoder,
    get_custom_decoders,
)

__all__ = [
    "DECODER_LABELS",
    "SinterMwpmDecoder",
    "SinterUnionFindDecoder",
    "get_custom_decoders",
]
