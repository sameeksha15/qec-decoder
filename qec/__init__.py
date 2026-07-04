"""qec — a study of quantum error correction decoders on the surface code.

This package is organized so that each concern has an obvious home:

- ``qec.codes``    : circuit/code constructors.
- ``qec.noise``    : circuit-level noise models.
- ``qec.decoders`` : a uniform decoder interface over MWPM / union-find / BP+OSD.
- ``qec.harness``  : Monte-Carlo sampling, threshold estimation, and plotting.

The public surface is intentionally small; import from the submodules directly.
"""

__version__ = "0.1.0"
