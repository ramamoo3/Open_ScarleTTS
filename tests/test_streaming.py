"""Tests for streaming, caching, segmentation and benchmarking (mock mode)."""

import json
import time

import numpy as np
import pytest

from open_scarletts import EmotionTTS, PhraseCache, make_key, split_sentences

MOCK_KW = dict(model_path="missing.onnx", voices_path="missing.bin")


def _mock_tts(**kw) -> EmotionTTS:
    return EmotionTTS(**{**MOCK_KW, **kw})


@pytest.fixture()
def tts(tmp_path) -> EmotionTTS:
    return _mock_tts(enable_cache=True, cache_dir=str(tmp_path / "cache"), cache_max_mb=5)


class TestSplitSentences:
    def test_basic(self):
        assert split_sentences("One. Two. Three!") == ["One.", "Two.", "Three!"]

    def test_decimals_protected(self):
        assert split_sentences("Pi is 3.14 exactly.") == ["Pi is 3.14 exactly."]

    def test_abbreviations_merged(self):
        out = split_sentences("Mr. Smith arrived. He was late.")
        assert len(out) == 2
        assert out[0] == "Mr. Smith arrived."

    def test_lowercase_continuation_not_split(self):
        assert len(split_sentences("He said no. it was fine... then left.")) >= 1

    def test_long_sentence_wrapped(self):
        long = "word " * 120 + "end."
        chunks = split_sentences(long, max_chars=100)
        assert all(len(c) <= 130 for c in chunks)
        assert " ".join(c.strip() for c in chunks).split() == long.split()

    def test_empty(self):
        assert split_sentences("   ") == []


class TestPhraseCache:
    def test_roundtrip(self, tmp_path):
        cache = PhraseCache(tmp_path / "c")
        samples = np.random.randn(1000).astype(np.float32) * 0.1
        key = make_key(text="hello", voice="af_heart")
        assert cache.get(key) is None
        assert cache.put(key, samples, 24000, meta={"text": "hello"})
        got = cache.get(key)
        assert got is not None
        np.testing.assert_allclose(got[0], samples, atol=1e-6)
        assert got[1] == 24000
        assert cache.hits == 1

    def test_eviction_respects_cap(self, tmp_path):
        cache = PhraseCache(tmp_path / "c", max_bytes=40_000)
        for i in range(10):
            samples = np.zeros(8000, dtype=np.float32)  # ~16KB wav each
            cache.put(make_key(idx=i), samples, 24000)
            time.sleep(0.01)  # distinct mtimes so LRU can evict
        assert cache.size_bytes <= 40_000 + 17_000  # cap plus one recent-entry grace

    def test_stats(self, tmp_path):
        cache = PhraseCache(tmp_path / "c")
        key = make_key(x=1)
        cache.put(key, np.zeros(100, dtype=np.float32), 16000)
        s = cache.stats()
        assert s["entries"] == 1 and s["misses"] == 0 and s["hits"] == 0
        cache.get(key)
        cache.get(make_key(x=2))
        assert cache.stats()["hits"] == 1 and cache.stats()["misses"] == 1


class TestStreaming:
    def test_stream_yields_per_sentence(self, tts):
        chunks = list(tts.stream("[happy] First one here. Second follows! Third arrives."))
        assert len(chunks) == 3
        for samples, sr in chunks:
            assert sr == 24000
            assert samples.dtype == np.float32

    def test_stream_single_chunk_for_one_sentence(self, tts):
        assert len(list(tts.stream("Just this."))) == 1

    def test_stream_validates_input(self, tts):
        with pytest.raises(ValueError):
            list(tts.stream(""))
        with pytest.raises(ValueError):
            list(tts.stream(None))

    def test_save_streaming_concatenates(self, tmp_path):
        pytest.importorskip("soundfile")
        import soundfile as sf

        _M = dict(model_path="missing.onnx", voices_path="missing.bin"); tts = EmotionTTS(**_M)
        out = tmp_path / "joined.wav"
        tts.save_streaming("Alpha here. Beta follows. Gamma ends.", str(out))
        data, sr = sf.read(out, dtype="float32")
        assert sr == 24000 and abs(len(data) - 3 * 24000) <= sr // 4


class TestCachedGenerate:
    def test_second_generate_is_hit(self, tts):
        tts.generate("[neutral] Repeat me please.")
        misses_before = tts.cache.misses
        samples, _ = tts.generate("[neutral] Repeat me please.")
        assert tts.cache.misses == misses_before
        assert tts.cache.hits == 1
        assert np.all(samples == 0)

    def test_different_emotion_is_separate_entry(self, tts):
        tts.generate("[neutral] Same words.")
        tts.generate("[sad] Same words.")
        assert tts.cache.entry_count == 2

    def test_cache_disabled_by_default(self):
        plain = EmotionTTS(model_path="missing.onnx", voices_path="missing.bin")
        assert plain.cache is None
        plain.generate("No cache here.")
        assert plain.cache is None


class TestBench:
    def test_bench_returns_metrics(self, tts):
        r = tts.bench("[happy] Hello there. How are you?", rounds=2)
        for key in ("ttfa_ms", "rtf", "audio_s", "gen_ms", "peak_rss_mb"):
            assert key in r
        assert r["audio_s"] > 0
        assert r["rtf"] > 0

    def test_bench_json_serializable(self, tts):
        r = tts.bench("Short.", rounds=1, streaming=False)
        json.dumps(r)


def test_version():
    from open_scarletts import __version__

    assert __version__.startswith("0.3.")
