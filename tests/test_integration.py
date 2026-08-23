"""Integration tests: run against the real Kokoro assets when installed.

Skipped automatically when assets are absent (CI / fresh checkouts).
"""

import numpy as np
import pytest

from open_scarletts import EmotionTTS
from open_scarletts.assets import have_assets

pytestmark = pytest.mark.skipif(
    not have_assets(), reason="real model assets not downloaded (run scarletts-setup)"
)


@pytest.fixture(scope="module")
def real_tts() -> EmotionTTS:
    return EmotionTTS(enable_cache=True, cache_dir=".scarletts_test_cache")


def test_real_synthesis_is_not_silent(real_tts):
    samples, sr = real_tts.generate("[neutral] Integration test one two three.")
    assert not real_tts.is_mock
    assert sr == 24000
    assert len(samples) > sr // 2
    assert 0.01 < float(np.abs(samples).max()) <= 1.0


def test_real_whisper_is_quieter_than_happy(real_tts):
    happy, _ = real_tts.generate("[happy] The robot sings a happy song.")
    whisper, _ = real_tts.generate("[whisper] The robot whispers very softly.")
    rms = lambda x: float(np.sqrt(np.mean(x**2)))
    assert rms(whisper) < rms(happy)


def test_real_streaming_chunks(real_tts):
    chunks = list(real_tts.stream("[happy] First real sentence. Second follows! Third ends it."))
    assert len(chunks) == 3


def test_real_rtf_below_one(real_tts):
    r = real_tts.bench("[happy] Benchmark on the real model now.", rounds=2)
    assert 0 < r["rtf"] < 1.5
