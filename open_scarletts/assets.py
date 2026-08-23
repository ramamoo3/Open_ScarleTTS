"""Asset management: download and locate Kokoro model files for plug-and-play.

Assets live in a per-user cache directory (``~/.cache/open_scarletts`` by
default, respecting ``$SCARLETT_CACHE_DIR`` / ``$XDG_CACHE_HOME``) so any
project on the machine shares one copy.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger("open_scarletts")

ASSET_URLS = {
    "kokoro-v1.0.int8.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin",
}

MIN_SIZES = {  # sanity floors so we never "succeed" with an HTML error page
    "kokoro-v1.0.int8.onnx": 50_000_000,
    "voices-v1.0.bin": 20_000_000,
}


def cache_dir() -> Path:
    env = os.environ.get("SCARLETT_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
    base = Path(env) if env else Path.home() / ".cache"
    d = base / "open_scarletts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def asset_path(name: str) -> Path:
    return cache_dir() / name


def have_assets(model_path: str | Path | None = None, voices_path: str | Path | None = None) -> bool:
    """True when both assets exist either at the given paths or in the cache."""
    m_ok = model_path is not None and Path(model_path).exists()
    v_ok = voices_path is not None and Path(voices_path).exists()
    return (
        (m_ok or asset_path("kokoro-v1.0.int8.onnx").exists())
        and (v_ok or asset_path("voices-v1.0.bin").exists())
    )


def resolve_paths(model_path: str | None, voices_path: str | None) -> tuple[str, str]:
    """Prefer explicit paths; fall back to the shared cache directory.

    Cache fallback only applies to the canonical asset names (or empty
    values) so callers pointing at custom/placeholder files keep them.
    """
    def pick(value: str | None, default_name: str) -> str:
        if value:
            p = Path(value)
            if p.exists() or p.name != default_name:
                return str(p)
        return str(asset_path(default_name))

    return (
        pick(model_path, "kokoro-v1.0.int8.onnx"),
        pick(voices_path, "voices-v1.0.bin"),
    )


def download_asset(name: str, dest: Path | None = None, show_progress: bool = True) -> Path:
    """Download one asset with resume support and size verification."""
    if name not in ASSET_URLS:
        raise ValueError(f"unknown asset {name!r}")
    url = ASSET_URLS[name]
    dest = dest or asset_path(name)
    part = dest.with_suffix(dest.suffix + ".part")
    min_size = MIN_SIZES[name]

    expected_total = 0
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            expected_total = int(resp.headers.get("Content-Length") or 0)
    except Exception:  # noqa: BLE001 - some CDNs reject HEAD; resume still works
        pass

    if dest.exists() and dest.stat().st_size >= min_size:
        return dest

    resume_from = part.stat().st_size if part.exists() else 0
    if expected_total and resume_from >= expected_total:
        # stale/partial beyond repair; start over
        part.unlink(missing_ok=True)
        resume_from = 0

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    request = urllib.request.Request(url, headers=headers)

    def report(count: int, block: int, total: int) -> None:
        done = resume_from + count * block
        total = total + resume_from if total > 0 else expected_total
        if show_progress:
            pct = min(100.0, 100.0 * done / total) if total else 0.0
            print(f"\r  {name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)", end="", flush=True)

    try:
        with urllib.request.urlopen(request, timeout=60) as resp, open(part, "ab" if resume_from else "wb") as fh:
            shutil.copyfileobj(resp, fh, 1024 * 256)
            downloaded = fh.tell() + resume_from
    finally:
        if show_progress:
            print()

    if downloaded < min_size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded {name} looks truncated ({downloaded} bytes)")
    part.replace(dest)
    return dest


def cli_setup() -> int:
    """Console-script entry: scarletts-setup."""
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="scarletts-setup", description="Download Open_ScarleTTS model assets.")
    p.add_argument("--model", default=None)
    p.add_argument("--voices", default=None)
    args = p.parse_args()
    try:
        model, voices = ensure_assets(args.model, args.voices, auto_download=True)
    except Exception as exc:
        print(f"setup failed: {exc}", file=sys.stderr)
        return 1
    print("Ready:")
    print(f"  model : {model}")
    print(f"  voices: {voices}")
    return 0


def ensure_assets(
    model_path: str | None = None,
    voices_path: str | None = None,
    auto_download: bool = False,
    quiet: bool = False,
) -> tuple[str, str]:
    """Resolve final asset paths, downloading into the cache if allowed.

    Returns ``(model_path, voices_path)`` that exist on disk.
    Raises ``FileNotFoundError`` when missing and ``auto_download`` is off.
    """
    m_str, v_str = resolve_paths(model_path, voices_path)
    missing = [p for p in (m_str, v_str) if not Path(p).exists()]
    if not missing:
        return m_str, v_str

    if not auto_download:
        hint = "\n".join(f"  - {m}" for m in missing)
        raise FileNotFoundError(
            f"missing model assets:\n{hint}\n"
            "run 'scarletts --setup' (or pass auto_download=True) to fetch them\n"
        )

    for name, existing in (("kokoro-v1.0.int8.onnx", m_str), ("voices-v1.0.bin", v_str)):
        if not Path(existing).exists():
            target = asset_path(name)
            logger.info("Downloading %s ...", name)
            download_asset(name, target, show_progress=not quiet)
    return resolve_paths(model_path, voices_path)
