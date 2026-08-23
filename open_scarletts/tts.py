"""Core EmotionTTS engine for Open_ScarleTTS.

A lightweight wrapper around Kokoro-ONNX that adds runtime emotion
styling via inline tags (e.g. "[happy] Hello!"), streaming synthesis,
a disk-backed phrase cache, and low-overhead NumPy DSP.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional dependency
    sf = None

try:
    from kokoro_onnx import Kokoro
except ImportError:  # pragma: no cover - optional dependency
    Kokoro = None

from .assets import DEFAULT_PRECISION, ensure_assets
from .cache import PhraseCache, make_key
from .speaker import play_stream, tee_to_file
from .textsplit import split_sentences

__all__ = ["EmotionTTS"]

logger = logging.getLogger("open_scarletts")

MOCK_SAMPLE_RATE = 24000
CACHE_VERSION = "scarletts-1"

_EMOTION_TAG_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*")


class EmotionTTS:
    """Emotion-styled text-to-speech engine backed by Kokoro-ONNX.

    Falls back to a safe mock mode (silence) when the model, voice
    assets, or required libraries are unavailable, so the class can
    always be constructed.
    """

    #: Default emotion profiles. ``speed`` feeds Kokoro's native speed
    #: control; ``pitch`` (>1 higher / <1 lower) is applied afterwards
    #: via resampling-based pitch shifting.
    DEFAULT_PROFILES: Dict[str, Dict[str, float]] = {
        "neutral": {"speed": 1.0, "pitch": 1.0},
        "happy": {"speed": 1.15, "pitch": 1.05},
        "sad": {"speed": 0.85, "pitch": 0.95},
        "angry": {"speed": 1.25, "pitch": 0.92},
        "whisper": {"speed": 0.90, "pitch": 1.02},
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        voices_path: Optional[str] = None,
        default_voice: str = "af_heart",
        lang: str = "en-us",
        emotion_profiles: Optional[Dict[str, Dict[str, float]]] = None,
        enable_cache: bool = False,
        cache_dir: str = ".scarletts_cache",
        cache_max_mb: int = 100,
        auto_download: bool = False,
        precision: str = DEFAULT_PRECISION,
    ) -> None:
        """Initialize the engine.

        Args:
            model_path: Path to the Kokoro ONNX model file.
            voices_path: Path to the voices ``.bin`` file.
            default_voice: Voice name used when none is given per call.
            lang: Default language code passed to Kokoro.
            emotion_profiles: Optional extra/overriding emotion profiles
                merged on top of :data:`DEFAULT_PROFILES`. Each profile
                supports ``speed`` and ``pitch`` floats (the legacy key
                ``pitch_mod`` is accepted as an alias for ``pitch``).
            enable_cache: Enable the disk-backed phrase cache.
            cache_dir: Cache directory location.
            cache_max_mb: Maximum cache size in megabytes before LRU eviction.

        Raises:
            ValueError: If ``default_voice``, ``model_path`` or
                ``voices_path`` is empty.
        """
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[EmotionTTS] %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        if not default_voice or not default_voice.strip():
            raise ValueError("default_voice must be a non-empty string")
        if model_path is not None and not str(model_path).strip():
            raise ValueError("model_path must be a non-empty string")
        if voices_path is not None and not str(voices_path).strip():
            raise ValueError("voices_path must be a non-empty string")
        if precision not in ("fp16", "int8"):
            raise ValueError(f"precision must be 'fp16' or 'int8', got {precision!r}")
        model_path = model_path or f"kokoro-v1.0.{precision}.onnx"
        voices_path = voices_path or "voices-v1.0.bin"

        self.model_path = model_path
        self.voices_path = voices_path
        self.default_voice = default_voice
        self.precision = precision
        self.lang = lang
        self.kokoro = None

        # default-family request keeps an alternate-precision fallback handy
        if Path(model_path).name.startswith("kokoro-v1.0."):
            self._fallback_model = "kokoro-v1.0.int8.onnx" if precision == "fp16" else "kokoro-v1.0.fp16.onnx"
        else:
            self._fallback_model = None
        try:
            resolved_model, resolved_voices = ensure_assets(
                model_path, voices_path, auto_download=auto_download, precision=precision
            )
            self.model_path, self.voices_path = resolved_model, resolved_voices
        except FileNotFoundError:
            pass  # _init_model logs mock-mode guidance below
        except Exception as exc:  # noqa: BLE001 - network errors
            logger.warning("Asset auto-download failed (%s); continuing.", exc)

        self.emotion_profiles: Dict[str, Dict[str, float]] = {
            name: dict(profile) for name, profile in self.DEFAULT_PROFILES.items()
        }
        if emotion_profiles:
            for name, profile in emotion_profiles.items():
                self.register_emotion(name, **profile)

        self.cache: Optional[PhraseCache] = None
        if enable_cache:
            try:
                self.cache = PhraseCache(cache_dir, max_bytes=cache_max_mb * 1024 * 1024)
                logger.info("Phrase cache enabled at '%s' (max %d MB).", cache_dir, cache_max_mb)
            except Exception as exc:  # noqa: BLE001 - filesystem errors
                logger.warning("Failed to open phrase cache (%s); continuing uncached.", exc)

        self._init_model()

    @classmethod
    def from_config(cls, path: str) -> "EmotionTTS":
        """Build an instance from a TOML config file.

        Supported keys under the top level or a ``[tts]`` table mirror
        the constructor keyword arguments.
        """
        try:
            import tomllib  # py>=3.11
        except ImportError:  # pragma: no cover
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ImportError as exc:
                raise RuntimeError(
                    "TOML support requires Python >=3.11 or the 'tomli' package."
                ) from exc
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        kwargs = dict(data.get("tts", data))
        known = {
            "model_path", "voices_path", "default_voice", "lang",
            "enable_cache", "cache_dir", "cache_max_mb",
        }
        unknown = set(kwargs) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        profiles = kwargs.pop("_profiles", None)
        return cls(emotion_profiles=profiles, **kwargs)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        """Load the Kokoro model if possible; otherwise enter mock mode."""
        if Kokoro is None or sf is None:
            missing = [
                name
                for name, module in (("kokoro-onnx", Kokoro), ("soundfile", sf))
                if module is None
            ]
            logger.warning(
                "Missing libraries (%s); running in mock mode.",
                ", ".join(missing),
            )
            return

        if os.path.exists(self.model_path) and os.path.exists(self.voices_path):
            try:
                self.kokoro = Kokoro(self.model_path, self.voices_path)
                logger.info("Loaded model from '%s'", self.model_path)
            except Exception as exc:  # noqa: BLE001 - third-party loader
                alt = self._fallback_asset()
                if alt is not None:
                    logger.warning("Loading '%s' failed (%s); trying '%s'.", self.model_path, exc, alt)
                    try:
                        self.kokoro = Kokoro(str(alt), self.voices_path)
                        self.model_path = str(alt)
                        logger.info("Loaded fallback model '%s'", self.model_path)
                        return
                    except Exception as exc2:  # noqa: BLE001
                        logger.warning("Fallback also failed (%s); running in mock mode.", exc2)
                else:
                    logger.warning(
                        "Failed to initialize Kokoro-ONNX (%s); running in mock mode.",
                        exc,
                    )
        else:
            logger.warning(
                "Model or voice files missing (model: '%s', voices: '%s'); "
                "running in mock mode.",
                self.model_path,
                self.voices_path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available_emotions(self) -> Tuple[str, ...]:
        """Names of all registered emotions."""
        return tuple(sorted(self.emotion_profiles))

    @property
    def is_mock(self) -> bool:
        """True when running without the real synthesis backend."""
        return self.kokoro is None

    def register_emotion(self, name: str, speed: float = 1.0, pitch: float = 1.0, **kwargs) -> None:
        """Register or override an emotion profile at runtime.

        Args:
            name: Tag users will write in brackets, e.g. ``"[excited]"``.
            speed: Speaking rate multiplier fed to Kokoro (0.5 - 2.0).
            pitch: Pitch shift multiplier from resampling (0.5 - 2.0).
            **kwargs: Legacy aliases; ``pitch_mod`` is treated as ``pitch``.

        Raises:
            ValueError: If ``name`` is empty, not a valid tag, or the
                modifiers are outside their allowed ranges.
        """
        pitch = kwargs.pop("pitch_mod", pitch)
        if kwargs:
            raise TypeError(f"Unknown profile options: {sorted(kwargs)}")
        clean_name = str(name).strip().lower()
        if not clean_name or not re.fullmatch(r"[a-z][a-z0-9_-]*", clean_name):
            raise ValueError(f"Invalid emotion name: {name!r}")
        if not 0.5 <= float(speed) <= 2.0:
            raise ValueError(f"speed must be within [0.5, 2.0], got {speed}")
        if not 0.5 <= float(pitch) <= 2.0:
            raise ValueError(f"pitch must be within [0.5, 2.0], got {pitch}")
        self.emotion_profiles[clean_name] = {"speed": float(speed), "pitch": float(pitch)}
        logger.debug("Registered emotion '%s' (speed=%s, pitch=%s)", clean_name, speed, pitch)

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        output_file: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """Generate audio for ``text``, applying any leading emotion tag.

        Args:
            text: Text to synthesize; may start with a tag such as
                ``"[happy]"``. Unknown tags are stripped with a warning.
            voice: Voice name overriding ``default_voice`` for this call.
            output_file: Optional path; audio is written there via
                ``soundfile`` when available.
            lang: Language code overriding the instance default.

        Returns:
            Tuple of ``(samples, sample_rate)`` as float32 in [-1, 1].

        Raises:
            ValueError: If ``text`` is empty after tag parsing.
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text).__name__}")

        emotion, clean_text = self._parse_emotion(text)
        if not clean_text:
            raise ValueError("text contains no speakable content")

        samples, sample_rate = self._synthesize(emotion, clean_text, voice, lang)
        self._maybe_write(output_file, samples, sample_rate)
        return samples, sample_rate

    def stream(
        self,
        text: str,
        voice: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Iterator[Tuple[np.ndarray, int]]:
        """Yield ``(samples, sample_rate)`` per sentence as they are synthesized.

        The leading emotion tag applies to the whole utterance. Feed the
        result into :func:`open_scarletts.speaker.play_stream` for gapless
        playback that starts before synthesis finishes.
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text).__name__}")

        emotion, clean_text = self._parse_emotion(text)
        if not clean_text:
            raise ValueError("text contains no speakable content")

        for sentence in split_sentences(clean_text):
            yield self._synthesize(emotion, sentence, voice, lang)

    def speak_streaming(self, text: str, **kwargs) -> None:
        """Synthesize and play chunk-by-chunk with gapless output."""
        play_stream(self.stream(text, **kwargs))

    def save_streaming(self, text: str, path: str, **kwargs) -> None:
        """Synthesize chunk-by-chunk and write all audio to ``path``."""
        for _ in tee_to_file(self.stream(text, **kwargs), path):
            pass

    def bench(
        self,
        text: str,
        rounds: int = 3,
        streaming: bool = True,
    ) -> Dict[str, float]:
        """Measure time-to-first-audio, RTF and peak RSS on this machine.

        Returns a dict with ``ttfa_ms``, ``rtf``, ``audio_s``,
        ``gen_ms``, ``peak_rss_mb`` and ``cache_hits``/``cache_misses``.
        """
        import resource
        import sys

        results: Dict[str, float] = {}
        gen_total = 0.0
        audio_total = 0.0

        first_round = True
        for r in range(max(1, rounds)):
            start = time.perf_counter()
            ttfa: Optional[float] = None
            produced = 0
            sr_ref = MOCK_SAMPLE_RATE
            source = self.stream(text) if streaming else iter([self.generate(text)])
            for samples, sr in source:
                if ttfa is None:
                    ttfa = (time.perf_counter() - start) * 1000.0
                produced += len(samples)
                sr_ref = sr
            elapsed = time.perf_counter() - start
            if ttfa is None:
                ttfa = elapsed * 1000.0
            gen_total += elapsed
            audio_total += produced / sr_ref
            if first_round:
                results["ttfa_ms"] = ttfa
                first_round = False

        results["rtf"] = gen_total / audio_total if audio_total > 0 else float("inf")
        results["audio_s"] = audio_total / max(1, rounds)
        results["gen_ms"] = gen_total * 1000.0 / max(1, rounds)
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        results["peak_rss_mb"] = raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024
        if self.cache is not None:
            stats = self.cache.stats()
            results["cache_hits"] = stats["hits"]
            results["cache_misses"] = stats["misses"]
        return results

    def close(self) -> None:
        """Release the ONNX session, if one is loaded."""
        session = getattr(self.kokoro, "session", None)
        if session is not None:
            try:
                session.__del__()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        self.kokoro = None

    def __enter__(self) -> "EmotionTTS":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        emotion: str,
        text: str,
        voice: Optional[str],
        lang: Optional[str],
    ) -> Tuple[np.ndarray, int]:
        """Single-chunk synthesis pipeline: cache lookup, backend, DSP."""
        resolved_voice = voice or self.default_voice
        resolved_lang = lang or self.lang
        profile = self.emotion_profiles[emotion]

        key = None
        if self.cache is not None:
            key = make_key(
                version=CACHE_VERSION,
                text=text,
                emotion=emotion,
                voice=resolved_voice,
                lang=resolved_lang,
                speed=profile["speed"],
                pitch=profile["pitch"],
                model=os.path.basename(str(self.model_path)),
            )
            cached = self.cache.get(key)
            if cached is not None:
                logger.debug("Cache hit for %r", text[:40])
                return cached

        if self.is_mock:
            logger.info("[MOCK] %s style: %r", emotion.upper(), text)
            samples = np.zeros(MOCK_SAMPLE_RATE, dtype=np.float32)
            sample_rate = MOCK_SAMPLE_RATE
        else:
            pitch = profile["pitch"]
            # Pitch shifting by resampling also compresses/expands time by
            # 1/pitch; compensate via Kokoro's native speed so the overall
            # pace still matches the profile's intended speed.
            kokoro_speed = round(profile["speed"] / pitch, 4)
            try:
                samples, sample_rate = self.kokoro.create(
                    text,
                    voice=resolved_voice,
                    speed=kokoro_speed,
                    lang=resolved_lang,
                )
                samples = np.asarray(samples, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001 - third-party synthesizer
                logger.error("Kokoro generation failed (%s); returning silence.", exc)
                samples = np.zeros(MOCK_SAMPLE_RATE, dtype=np.float32)
                sample_rate = MOCK_SAMPLE_RATE

            samples = self._apply_dsp(samples, emotion, profile["pitch"])

        if key is not None:
            self.cache.put(
                key,
                samples,
                sample_rate,
                meta={"text": text[:200], "emotion": emotion, "voice": resolved_voice},
            )
        return samples, sample_rate

    def _fallback_asset(self):
        """Locate an alternate-precision Kokoro file (cache dir first)."""
        name = getattr(self, "_fallback_model", None)
        if not name:
            return None
        for cand in (Path(self.model_path).parent / name, Path.home() / ".cache" / "open_scarletts" / name):
            if cand.exists():
                return cand
        return None

    def _parse_emotion(self, text: str) -> Tuple[str, str]:
        """Extract a leading ``[tag]`` and return ``(emotion, clean_text)``.

        Known tags select the profile. Unknown tags are stripped and a
        warning is logged instead of being spoken aloud.
        """
        match = _EMOTION_TAG_RE.match(text)
        if match:
            emotion = match.group(1).strip().lower()
            clean_text = text[match.end():].strip()
            if emotion in self.emotion_profiles:
                return emotion, clean_text
            logger.warning(
                "Unknown emotion tag '[%s]' ignored. Supported: %s",
                emotion,
                ", ".join(self.available_emotions),
            )
            return "neutral", clean_text
        return "neutral", text.strip()

    def _apply_dsp(self, samples: np.ndarray, emotion: str, pitch: float) -> np.ndarray:
        """Apply pitch shifting and emotion-specific DSP, then clip."""
        if abs(pitch - 1.0) > 1e-3:
            samples = self._pitch_shift(samples, pitch)
        if emotion == "whisper":
            samples = self._simulate_whisper(samples)
        return np.clip(samples, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _pitch_shift(samples: np.ndarray, factor: float) -> np.ndarray:
        """Shift pitch by ``factor`` (>1 raises pitch) via resampling.

        Note this also scales playback duration by ``1/factor``; callers
        compensate through Kokoro's native speed parameter.
        """
        n = len(samples)
        if n < 2 or abs(factor - 1.0) < 1e-3:
            return samples
        new_len = max(2, int(round(n / factor)))
        x_old = np.linspace(0.0, 1.0, n)
        x_new = np.linspace(0.0, 1.0, new_len)
        return np.interp(x_new, x_old, samples).astype(np.float32)

    @staticmethod
    def _simulate_whisper(samples: np.ndarray) -> np.ndarray:
        """Simulate a whisper: amplitude gating plus breathy white noise."""
        damped = samples * 0.4
        noise = np.random.normal(0.0, 0.005, samples.shape).astype(np.float32)
        return damped + noise

    @staticmethod
    def _maybe_write(output_file: Optional[str], samples: np.ndarray, sample_rate: int) -> None:
        """Write ``samples`` to ``output_file`` if requested and possible."""
        if not output_file:
            return
        if sf is None:
            logger.warning("soundfile not installed; could not save '%s'.", output_file)
            return
        try:
            sf.write(output_file, samples, sample_rate)
        except Exception as exc:  # noqa: BLE001 - filesystem/audio errors
            logger.error("Failed to save audio file '%s': %s", output_file, exc)
