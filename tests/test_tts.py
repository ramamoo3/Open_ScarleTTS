"""Tests for Open_ScarleTTS (run without model assets; mock mode)."""

import numpy as np
import pytest

from open_scarletts import EmotionTTS, __version__


MOCK_KW = dict(model_path="missing.onnx", voices_path="missing.bin")


@pytest.fixture()
def tts() -> EmotionTTS:
    return _mock_tts()


def _mock_tts(**kw) -> EmotionTTS:
    return EmotionTTS(**{**MOCK_KW, **kw})


def sine(freq: int = 440, seconds: float = 0.5, rate: int = 24000) -> np.ndarray:
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestParseEmotion:
    def test_known_tag(self, tts):
        emotion, text = tts._parse_emotion("[happy] Good morning!")
        assert emotion == "happy"
        assert text == "Good morning!"

    def test_case_and_whitespace_insensitive(self, tts):
        emotion, text = tts._parse_emotion("  [ HAPPY ]  Hi there")
        assert emotion == "happy"
        assert text == "Hi there"

    def test_unknown_tag_is_stripped_not_spoken(self, tts, caplog):
        with caplog.at_level("WARNING", logger="open_scarletts"):
            emotion, text = tts._parse_emotion("[excited] Let's go!")
        assert emotion == "neutral"
        # Regression: the tag used to be left in the spoken text.
        assert text == "Let's go!"
        assert any("excited" in record.message for record in caplog.records)

    def test_no_tag_passthrough(self, tts):
        emotion, text = tts._parse_emotion("Plain speech.")
        assert emotion == "neutral"
        assert text == "Plain speech."


class TestProfiles:
    def test_register_emotion(self, tts):
        tts.register_emotion("excited", speed=1.3, pitch=1.1)
        assert "excited" in tts.available_emotions
        emotion, _ = tts._parse_emotion("[excited] Yes!")
        assert emotion == "excited"
        assert tts.emotion_profiles["excited"] == {"speed": 1.3, "pitch": 1.1}

    def test_register_accepts_legacy_pitch_mod(self, tts):
        tts.register_emotion("legacy", pitch_mod=1.2)
        assert tts.emotion_profiles["legacy"]["pitch"] == 1.2

    def test_register_rejects_bad_values(self, tts):
        with pytest.raises(ValueError):
            tts.register_emotion("bad name!")
        with pytest.raises(ValueError):
            tts.register_emotion("fast", speed=5.0)
        with pytest.raises(ValueError):
            tts.register_emotion("high", pitch=9.9)
        with pytest.raises(TypeError):
            tts.register_emotion("oops", bogus=1.0)

    def test_init_profiles_merge(self):
        tts = EmotionTTS(model_path="missing.onnx", voices_path="missing.bin", emotion_profiles={"sleepy": {"speed": 0.7}})
        assert tts.emotion_profiles["sleepy"] == {"speed": 0.7, "pitch": 1.0}
        assert "happy" in tts.available_emotions

    def test_default_profiles_not_mutated(self, tts):
        tts.register_emotion("temp", speed=1.9)
        assert "temp" not in EmotionTTS.DEFAULT_PROFILES


class TestGenerateMock:
    def test_mock_returns_silence(self, tts):
        samples, rate = tts.generate("[sad] Oh no.")
        assert tts.is_mock
        assert rate == 24000
        assert samples.dtype == np.float32
        assert samples.shape == (rate,)
        assert np.all(samples == 0)

    def test_empty_text_raises(self, tts):
        with pytest.raises(ValueError):
            tts.generate("")
        with pytest.raises(ValueError):
            tts.generate("[happy]   ")
        with pytest.raises(ValueError):
            tts.generate(None)

    def test_output_file_written(self, tmp_path):
        pytest.importorskip("soundfile")
        _M = dict(model_path="missing.onnx", voices_path="missing.bin"); tts = EmotionTTS(**_M)
        out = tmp_path / "out.wav"
        tts.generate("Hello file.", output_file=str(out))
        assert out.stat().st_size > 44

    def test_context_manager(self, tts):
        with tts as engine:
            assert engine is tts
        assert tts.is_mock


class TestDSP:
    def test_pitch_shift_changes_length(self, tts):
        x = sine()
        up = tts._pitch_shift(x, 2.0)
        down = tts._pitch_shift(x, 0.5)
        assert len(up) == len(x) // 2
        assert len(down) == len(x) * 2

    def test_pitch_shift_identity(self, tts):
        x = sine()
        assert tts._pitch_shift(x, 1.0) is x

    def test_whisper_is_quieter(self, tts):
        x = sine()
        whispered = tts._simulate_whisper(x)
        assert whispered.shape == x.shape
        assert np.sqrt(np.mean(whispered**2)) < np.sqrt(np.mean(x**2))

    def test_dsp_clips_to_valid_range(self, tts):
        loud = (sine() * 50).astype(np.float32)
        out = tts._apply_dsp(loud, "angry", pitch=0.92)
        assert out.dtype == np.float32
        assert np.max(np.abs(out)) <= 1.0

    def test_neutral_dsp_is_lossless(self, tts):
        x = sine()
        out = tts._apply_dsp(x.copy(), "neutral", pitch=1.0)
        np.testing.assert_allclose(out, x)


def test_version():
    assert isinstance(__version__, str)
