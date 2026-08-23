"""Benchmark harness: TTFA, RTF and peak RSS for Open_ScarleTTS."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from . import EmotionTTS, __version__

DEFAULT_CASES = {
    "short": "[neutral] Yes.",
    "sentence": "[happy] I am so excited to show you what this little board can do!",
    "paragraph": (
        "[sad] It was a long winter, and the station had been silent for months. "
        "Then one morning, the radio crackled to life. [happy] Someone was out there! "
        "[whisper] But who would answer at this hour?"
    ),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="scarletts-bench", description=__doc__)
    parser.add_argument("--model", default="kokoro-v1.0.int8.onnx")
    parser.add_argument("--voices", default="voices-v1.0.bin")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--no-stream", action="store_true", help="Benchmark non-streaming generate().")
    parser.add_argument("--cache-dir", default=None, help="Enable phrase cache in this directory.")
    parser.add_argument("--json", default=None, help="Write results as JSON to this file.")
    args = parser.parse_args(argv)

    tts = EmotionTTS(
        model_path=args.model,
        voices_path=args.voices,
        default_voice=args.voice,
        enable_cache=bool(args.cache_dir),
        cache_dir=args.cache_dir or ".scarletts_bench_cache",
    )

    print(f"scarletts-bench {__version__} | mock={tts.is_mock} | streaming={not args.no_stream}")
    print(f"{'case':<10} {'ttfa_ms':>9} {'rtf':>7} {'audio_s':>8} {'gen_ms':>8} {'rss_mb':>7}")

    all_results = {}
    for name, text in DEFAULT_CASES.items():
        r = tts.bench(text, rounds=args.rounds, streaming=not args.no_stream)
        all_results[name] = {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
        print(
            f"{name:<10} {r['ttfa_ms']:>9.1f} {r['rtf']:>7.3f} "
            f"{r['audio_s']:>8.2f} {r['gen_ms']:>8.1f} {r['peak_rss_mb']:>7.1f}"
        )

    if tts.cache is not None:
        all_results["cache"] = tts.cache.stats()
        print(f"cache: {all_results['cache']}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"engine": "open_scarletts-tier0", "mock": tts.is_mock, "results": all_results}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
