# Open_ScarleTTS 🎙️

Open_ScarleTTS is an open-source, ultra-lightweight, and highly expressive Text-To-Speech (TTS) engine designed specifically for edge systems (like the Raspberry Pi 5). 

It is built on top of the excellent **Kokoro-ONNX** model, executing an INT8 quantized variant to achieve extremely low RAM consumption and fast CPU synthesis while delivering natural, human-like voice quality.

## Features

- **Expressive Emotion Styling:** Inject runtime emotion shifts like `[happy]`, `[sad]`, `[angry]`, and `[whisper]` inline with your LLM text output.
- **Edge Optimized:** Designed to run entirely on-board memory-constrained devices.
- **Lightweight DSP Enhancements:** Uses low-overhead NumPy DSP modifications (such as volume damping and noise injection for whispers, speed manipulation) to shape expression without bloating dependencies.
- **Robust Fallbacks:** Safe mock mode fallback ensures the library can load and run cleanly even without model assets present.

## Installation

Ensure you have your system audio libraries installed (such as ALSA or PortAudio if on Linux/Raspberry Pi). Then clone the repository and install it in editable mode:

```bash
git clone https://github.com/ramamoo3/Open_ScarleTTS.git
cd Open_ScarleTTS
pip install -e .
```

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
```

## Supported Emotions

- `[neutral]` - Default speech rate and volume.
- `[happy]` - Elevates speed and brightens articulation.
- `[sad]` - Drastically reduces speed, introducing somber pauses.
- `[angry]` - Spikes speech speed with lower pitch emphasis.
- `[whisper]` - Replaces typical audio volume with soft amplitude gating and a low-intensity white noise backdrop to recreate a real whisper.
