"""Decoder latency benchmarking.

Methodology (documented because fair timing is subtle):

- Every decoder is timed through the **same call path**: the bit-packed
  ``sinter.CompiledDecoder.decode_shots_bit_packed`` interface that the
  accuracy runs also use. What we measure is therefore "the throughput of each
  library's standard batch API as actually used", including any per-shot Python
  overhead a wrapper needs — because the accuracy numbers were produced through
  that exact path too.
- Syndromes are sampled **once** per circuit (fixed seed) and every decoder
  decodes the *same* batch.
- A **warm-up** decode runs first (JIT/caches/allocations), then the batch is
  timed ``repeats`` times and we report the **median** per-shot time, which is
  robust to OS scheduling noise on a shared machine.
- Timing uses ``time.perf_counter`` around the whole batch, divided by the
  number of shots: a per-shot *average*, appropriate for throughput comparison
  (tail latencies would need a different experiment).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import sinter
import stim


@dataclass(frozen=True)
class LatencyResult:
    """Median per-shot decode time for one (decoder, circuit) pair."""

    decoder_name: str
    seconds_per_shot: float
    shots: int
    repeats: int

    @property
    def us_per_shot(self) -> float:
        return self.seconds_per_shot * 1e6


def benchmark_decoder_latency(
    circuit: stim.Circuit,
    decoders: dict[str, sinter.Decoder],
    *,
    num_shots: int = 2_000,
    warmup_shots: int = 100,
    repeats: int = 5,
    seed: int = 2026_02_06,
) -> list[LatencyResult]:
    """Time each decoder on an identical batch of sampled syndromes.

    Args:
        circuit: A noisy circuit with detectors and observables (as produced by
            the ``qec.codes`` builders).
        decoders: Name -> ``sinter.Decoder`` registry (see
            :func:`qec.decoders.get_custom_decoders`).
        num_shots: Batch size to time. Thousands are enough for a stable
            per-shot average; the batch is decoded ``repeats`` times.
        warmup_shots: Shots decoded once before timing starts.
        repeats: Timed repetitions; the median is reported.
        seed: Sampler seed, fixed so every decoder sees identical syndromes and
            reruns are reproducible.

    Returns:
        One :class:`LatencyResult` per decoder, in input order.
    """
    if num_shots < 1 or warmup_shots < 0 or repeats < 1:
        raise ValueError("num_shots >= 1, warmup_shots >= 0, repeats >= 1 required")

    sampler = circuit.compile_detector_sampler(seed=seed)
    detection_events = sampler.sample(
        shots=num_shots + warmup_shots, bit_packed=True
    )
    warmup_batch = detection_events[:warmup_shots]
    timed_batch = detection_events[warmup_shots:]

    dem = circuit.detector_error_model(decompose_errors=True)

    results: list[LatencyResult] = []
    for name, decoder in decoders.items():
        compiled = decoder.compile_decoder_for_dem(dem=dem)

        if warmup_shots:
            compiled.decode_shots_bit_packed(
                bit_packed_detection_event_data=warmup_batch
            )

        durations: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            compiled.decode_shots_bit_packed(
                bit_packed_detection_event_data=timed_batch
            )
            durations.append(time.perf_counter() - start)

        results.append(
            LatencyResult(
                decoder_name=name,
                seconds_per_shot=float(np.median(durations)) / num_shots,
                shots=num_shots,
                repeats=repeats,
            )
        )
    return results
