# Raspberry Pi 5 Deployment Guide

Everything you need to take this repo from clone to speaking on a Pi 5
(4× Cortex-A76, 4–8 GB RAM) alongside an LLM.

## 1. System prep

```bash
sudo apt update && sudo apt install -y python3-dev python3-venv portaudio19-dev libasound2-dev git
```

> Use a real Python ≥3.10 if possible (`pyenv` or Raspberry Pi OS Bookworm's
> `python3.11`). Python 3.9 works but older wheels are hit-or-miss.

## 2. Install

```bash
git clone https://github.com/ramamoo3/Open_ScarleTTS.git
cd Open_ScarleTTS
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
scarletts-setup          # downloads ~163 MB (fp16) once → ~/.cache/open_scarletts
scarletts "Hello from the Raspberry Pi!"
```

If the fp16 model fails to load on your ORT build, the engine automatically
falls back to int8; you can also force it:

```bash
scarletts --precision int8 "Safety first."
```

## 3. Benchmark on YOUR board (please share numbers!)

```bash
scarletts-bench --precision fp16 --rounds 4 --json pi5_fp16.json
scarletts-bench --precision int8 --rounds 4 --json pi5_int8.json
```

Report `ttfa_ms` / `rtf` / `peak_rss_mb`. Reference points measured on an
Apple-Silicon Mac: fp16 RTF ≈ 0.23, int8 ≈ 0.43, RSS ≈ 330 MB. Expect the Pi5
to land somewhere between 2–6× slower; RTF < 1 is production-usable thanks to
streaming (playback never stalls once started).

## 4. Tuning for co-residency with an LLM

```bash
# Cap ONNX Runtime threads so the LLM keeps cores:
export OMP_NUM_THREADS=2
export ORT_NUM_THREADS=2   # honored by kokoro-onnx >= 0.4

# Persistent phrase cache = near-zero latency for repeated prompts:
scarletts --cache-dir ~/.scarletts_cache --cache-max-mb 200 ...
```

Memory budget target: TTS ≤ 500 MB typical, leaving the rest of a 2 GB
allocation for caches/headroom and ~6 GB for the LLM on an 8 GB board.

## 5. Streaming LLM pattern

```python
from open_scarletts import EmotionTTS

tts = EmotionTTS(enable_cache=True, cache_dir="~/.scarletts")

for chunk in llm_stream():                 # sentence-level text chunks
    tts.speak_streaming(f"[happy] {chunk}")  # speaks while next chunk generates
```

## Known unknowns (help fill these)

- [ ] fp16 vs int8 speed ranking on A76 (Mac showed fp16 2× faster)
- [ ] Sustained-load thermals during long renders
- [ ] USB mic/speaker latency via ALSA vs PulseAudio
