"""Command-line interface for Open_ScarleTTS."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from . import EmotionTTS, __version__

DEFAULT_MODEL = "kokoro-v1.0.int8.onnx"
DEFAULT_VOICES = "voices-v1.0.bin"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scarletts",
        description="Emotion-styled text-to-speech powered by Kokoro-ONNX.",
        epilog='Example: scarletts "[happy] Good morning!" -o hello.wav',
    )
    parser.add_argument("text", nargs="*", help='Text to speak; may start with an emotion tag like "[happy]".')
    parser.add_argument("-m", "--model", default=None, help=f"Path to the ONNX model (default: kokoro-v1.0.<precision>.onnx).")
    parser.add_argument("--voices", default=None, help=f"Path to the voices .bin file (default: voices-v1.0.bin).")
    parser.add_argument("-v", "--voice", default="af_heart", help="Voice name (default: af_heart).")
    parser.add_argument("--lang", default="en-us", help="Language code (default: en-us).")
    parser.add_argument("-o", "--output", help="Write audio to this WAV/FLAC file instead of playing it.")
    parser.add_argument("-s", "--stream", action="store_true", help="Stream sentence-by-sentence for gapless early playback.")
    parser.add_argument("--cache-dir", default=None, help="Enable the disk phrase cache in this directory.")
    parser.add_argument("--cache-max-mb", type=int, default=100, help="Phrase cache size cap in MB (default 100).")
    parser.add_argument("--setup", action="store_true", help="Download model assets to the shared cache and exit.")
    parser.add_argument("-y", "--auto-download", action="store_true", help="Download missing assets automatically on first use.")
    parser.add_argument("--precision", choices=["fp16", "int8"], default=None, help="Kokoro model variant (default: fp16, falls back to int8).")
    parser.add_argument("-l", "--list-emotions", action="store_true", help="List available emotions and exit.")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.setup:
        from .assets import ensure_assets

        model, voices = ensure_assets(args.model, args.voices, auto_download=True,
                                      precision=args.precision or "fp16")
        print("Ready:")
        print(f"  model : {model}")
        print(f"  voices: {voices}")
        if not args.text:
            return 0

    tts = EmotionTTS(
        model_path=args.model,
        voices_path=args.voices,
        default_voice=args.voice,
        lang=args.lang,
        enable_cache=bool(args.cache_dir),
        cache_dir=args.cache_dir or ".scarletts_cache",
        cache_max_mb=args.cache_max_mb,
        auto_download=args.auto_download,
        precision=args.precision or "fp16",
    )

    if args.list_emotions:
        for emotion in tts.available_emotions:
            profile = tts.emotion_profiles[emotion]
            print(f"  [{emotion:<8} speed={profile['speed']:<5} pitch={profile['pitch']}")
        return 0

    text = " ".join(args.text).strip()
    if not text:
        print("error: no text provided (pass text or use --list-emotions)", file=sys.stderr)
        return 2

    import os

    if args.stream:
        if args.output:
            tts.save_streaming(text, args.output, voice=args.voice)
        else:
            tts.speak_streaming(text, voice=args.voice)
        if args.output:
            print(f"Saved streamed audio to '{args.output}'")
        return 0

    samples, sample_rate = tts.generate(text, output_file=args.output)
    if args.output:
        if os.path.exists(args.output):
            print(f"Saved audio to '{args.output}'")
        else:
            print(f"error: could not write '{args.output}'", file=sys.stderr)
            return 1
    else:
        tts.play(samples, sample_rate)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
