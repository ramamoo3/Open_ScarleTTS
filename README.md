# Open_ScarleTTS 🎙️

Open_ScarleTTS is an open-source, ultra-lightweight, and highly expressive Text-To-Speech (TTS) engine designed specifically for edge systems (like the Raspberry Pi 5). 

It is built on top of the excellent **Kokoro-ONNX** model, executing an INT8 quantized variant to achieve extremely low RAM consumption and fast CPU synthesis while delivering natural, human-like voice quality.

## Features

- **Expressive Emotion Styling:** Inject runtime emotion shifts like `[happy]`, `[sad]`, `[angry]`, and `[whisper]` inline with your LLM text output.
- **Streaming Synthesis:** `tts.stream()` yields audio sentence-by-sentence; playback starts before synthesis finishes (gapless, flat memory).
- **Phrase Cache:** Disk-backed LRU cache — repeated phrases (toys, assistants) play back with ~0 ms latency.
- **Edge Optimized:** Designed to run entirely on-board memory-constrained devices. Full engine targets well under 500 MB RSS with the INT8 model, leaving the rest of a 2 GB TTS budget for caches or headroom.
- **Lightweight DSP Enhancements:** Uses low-overhead NumPy DSP (pitch shifting via resampling, volume damping, breathy noise injection for whispers) to shape expression without bloating dependencies.
- **Runtime Emotion Registration:** Add custom emotions with `tts.register_emotion("excited", speed=1.3, pitch=1.1)` — no source edits required.
- **CLI Included:** Speak or render files straight from the shell with the `scarletts` command (or `python -m open_scarletts`).
- **Benchmark Harness:** Measure time-to-first-audio, RTF and peak RSS on your exact hardware (`scarletts-bench`).
- **Robust Fallbacks:** Safe mock mode ensures the library loads and runs cleanly even without model assets present.
- **Proper Logging:** Structured `logging` output via the `open_scarletts` logger instead of raw prints.

## Installation

Ensure you have your system audio libraries installed (such as ALSA or PortAudio if on Linux/Raspberry Pi).

### Quickstart (plug and play)

```bash
git clone https://github.com/ramamoo3/Open_ScarleTTS.git
cd Open_ScarleTTS
pip install -e .
scarletts-setup          # downloads the ~140 MB Kokoro INT8 model once (~/.cache/open_scarletts)
scarletts "[happy] Hello world!"
```

Assets download once per machine into `~/.cache/open_scarletts` (override with `$SCARLETT_CACHE_DIR`), so every project shares them; if you run without assets present, the engine falls back to silent mock mode instead of crashing.

### Required Dependencies
Declares lightweight pins to avoid dependency bloat:
* `kokoro-onnx`
* `sounddevice`
* `numpy`
* `soundfile`

## Model & Voice Assets Setup

To utilize the actual deep learning synthesizer, you must download the Kokoro ONNX model and the voices bin file:

1. Download the INT8 quantized Kokoro model: `kokoro-v1.0.int8.onnx`
2. Download the voice bin file: `voices-v1.0.bin`
3. Place them in your project root or pass their paths directly to the `EmotionTTS` constructor.

### Memory Optimization for Raspberry Pi 🧠
To run with a minimal memory footprint, you can extract and save only your preferred voice embedding (e.g., `af_heart`) from the large `voices-v1.0.bin` file, discarding the unused voices. This can reduce the voice file size significantly.

## Usage

```python
from open_scarletts import EmotionTTS

# Initialize the engine
tts = EmotionTTS(
    model_path="kokoro-v1.0.int8.onnx",
    voices_path="voices-v1.0.bin",
    default_voice="af_heart"
)

# Generate emotion-styled speech
# The parser automatically detects tags at the beginning of the text
samples, sample_rate = tts.generate("[happy] I am absolutely thrilled to help you build this project!")

# Play audio
tts.play(samples, sample_rate)

# Or save directly to a file, and override voice/language per call
samples, sample_rate = tts.generate(
    "[whisper] It's a secret...",
    voice="af_bella",
    output_file="secret.wav",
)

# Streaming: speak sentence 1 while sentence 2 is still synthesizing
tts.speak_streaming("[happy] This is a long reply from your LLM! "
                    "It starts playing immediately. No more dead air.")

# Phrase cache: second identical phrase is ~0 ms
tts_cache = EmotionTTS(enable_cache=True, cache_dir="~/.scarletts", cache_max_mb=200)

# Register your own emotion at runtime (no source edits needed)
tts.register_emotion("excited", speed=1.3, pitch=1.1)
samples, sample_rate = tts.generate("[excited] Let's launch it right now!")
```

## Command Line

After installing the package:

```bash
scarletts "[happy] Good morning!"               # speak through speakers
scarletts "[sad] Goodbye..." -o goodbye.wav     # render to a WAV file instead
scarletts "[happy] Long story..." -s            # stream sentence-by-sentence
scarletts --cache-dir ~/.scarletts "Hello!"     # persistent phrase cache
scarletts --list-emotions                       # show all emotions and their profiles
scarletts-bench --json results.json             # TTFA / RTF / peak-RSS benchmark
```

## Supported Emotions

| Tag | Speed | Pitch | Character |
| --- | --- | --- | --- |
| `[neutral]` | 1.0 | 1.0 | Default speech rate and volume. |
| `[happy]` | 1.15 | 1.05 | Elevated speed, brighter articulation. |
| `[sad]` | 0.85 | 0.95 | Slowed pace with a somber tone. |
| `[angry]` | 1.25 | 0.92 | Fast, forceful delivery. |
| `[whisper]` | 0.90 | 1.02 | Soft amplitude gating plus a low-intensity noise bed for a breathy whisper. |

Unknown tags (e.g. `[excited]`) are stripped from the text and logged as a warning rather than spoken aloud — register them first with `register_emotion()` if you want them styled.

All DSP output is clipped to the `[-1, 1]` range to prevent speaker-damaging clipping on edge hardware.

## Ecosystem

Open_ScarleTTS is **Tier 0** of a three-track plan to get the best possible
on-device voice:

| Repo | Track | Status |
|---|---|---|
| `Open_ScarleTTS` (this repo) | Kokoro INT8 + streaming + caching | Production-ready |
| `scarletts-training` | Distill a 10–30M student from Kokoro-rendered data | Pipeline scaffold, CPU smoke-tested |
| `scarletts-stream` | Streaming codec-LM research bet | Prototype scaffold, CPU smoke-tested |

All tiers share a bench schema (`ttfa_ms`, `rtf`, `peak_rss_mb`); compare
results with `scarletts-training/tools/compare_tiers.py`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite runs entirely in mock mode; no model download is required.

## License

MIT — see [LICENSE](LICENSE).
