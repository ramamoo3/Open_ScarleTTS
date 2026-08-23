"""Disk-backed LRU phrase cache for synthesized audio."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional dependency
    sf = None


def make_key(**parts: Any) -> str:
    """Deterministic cache key from canonical JSON of the given parts."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class PhraseCache:
    """Stores synthesized utterances as WAV files with LRU byte-cap eviction.

    Thread-safe. Each entry is ``<key>.wav`` plus a ``<key>.json`` sidecar
    holding metadata (voice, emotion, text preview, sample rate).
    """

    def __init__(self, directory: str | Path, max_bytes: int = 100 * 1024 * 1024) -> None:
        if sf is None:
            raise RuntimeError("soundfile is required for PhraseCache")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Tuple[np.ndarray, int]]:
        path = self.directory / f"{key}.wav"
        with self._lock:
            if not path.exists():
                self.misses += 1
                return None
            try:
                samples, sr = sf.read(path, dtype="float32")
                path.touch()  # LRU touch
                self.hits += 1
                return samples.astype(np.float32), sr
            except Exception:
                self.misses += 1
                return None

    def put(
        self,
        key: str,
        samples: np.ndarray,
        sample_rate: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        wav_path = self.directory / f"{key}.wav"
        meta_path = self.directory / f"{key}.json"
        tmp_wav = self.directory / f"{key}.tmp.wav"  # .wav suffix so soundfile infers format
        try:
            with self._lock:
                sf.write(tmp_wav, samples, sample_rate, subtype="FLOAT")
                tmp_wav.replace(wav_path)
                meta_path.write_text(json.dumps(meta or {}, default=str))
                self._evict_locked()
            return True
        except Exception:
            tmp_wav.unlink(missing_ok=True)
            return False

    def _evict_locked(self) -> None:
        wavs = sorted(
            self.directory.glob("*.wav"), key=lambda p: p.stat().st_mtime
        )
        total = sum(p.stat().st_size for p in wavs)
        # LRU: oldest first, but always keep at least the newest entry.
        for path in wavs[:-1]:
            if total <= self.max_bytes:
                break
            total -= path.stat().st_size
            path.unlink(missing_ok=True)
            (self.directory / f"{path.stem}.json").unlink(missing_ok=True)

    @property
    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.directory.glob("*.wav"))

    @property
    def entry_count(self) -> int:
        return len(list(self.directory.glob("*.wav")))

    def clear(self) -> None:
        with self._lock:
            for p in list(self.directory.glob("*")):
                if p.suffix in (".wav", ".json") or p.name.endswith(".tmp.wav"):
                    p.unlink(missing_ok=True)

    def stats(self) -> Dict[str, Any]:
        return {
            "entries": self.entry_count,
            "bytes": self.size_bytes,
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
        }
