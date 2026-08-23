"""Continuous speaker playback for streaming synthesis."""

from __future__ import annotations

import logging
from typing import Iterable, Iterator, Tuple

import numpy as np

logger = logging.getLogger("open_scarletts")


def play_stream(chunks: Iterable[Tuple[np.ndarray, int]]) -> None:
    """Play an iterable of (samples, sample_rate) chunks with no inter-chunk gap.

    Opens a single output stream on the first chunk and pushes audio into
    it as synthesis continues, so playback starts before later chunks exist.
    """
    import sounddevice as sd

    iterator = iter(chunks)
    try:
        samples, sr = next(iterator)
    except StopIteration:
        return

    stream = sd.OutputStream(samplerate=int(sr), channels=1, dtype="float32")
    stream.start()
    try:
        stream.write(np.ascontiguousarray(samples, dtype=np.float32))
        for more_samples, more_sr in iterator:
            if int(more_sr) != int(sr):
                logger.warning("Sample rate changed (%s -> %s); reopening stream.", sr, more_sr)
                stream.stop()
                stream.close()
                sr = more_sr
                stream = sd.OutputStream(samplerate=int(sr), channels=1, dtype="float32")
                stream.start()
            stream.write(np.ascontiguousarray(more_samples, dtype=np.float32))
    finally:
        stream.stop()
        stream.close()


def tee_to_file(
    chunks: Iterator[Tuple[np.ndarray, int]],
    path: str,
) -> Iterator[Tuple[np.ndarray, int]]:
    """Pass-through generator that concatenates chunks into ``path`` when done."""
    import soundfile as sf

    collected: list = []
    sr = 24000
    for samples, sr in chunks:
        collected.append(samples)
        yield samples, sr
    if collected:
        sf.write(path, np.concatenate(collected), sr)
