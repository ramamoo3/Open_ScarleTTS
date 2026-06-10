import re
import os
import numpy as np

try:
    import soundfile as sf
    from kokoro_onnx import Kokoro
except ImportError:
    sf = None
    Kokoro = None

class EmotionTTS:
    def __init__(self, model_path="kokoro-v1.0.int8.onnx", voices_path="voices-v1.0.bin", default_voice="af_heart"):
        """
        Initializes the lightweight Kokoro TTS wrapper with emotion profile DSP styling.
        If model/voice files are not found, falls back to a safe mock mode.
        """
        self.model_path = model_path
        self.voices_path = voices_path
        self.default_voice = default_voice
        self.kokoro = None

        # DSP modifiers for different emotions
        # 'speed' is native to Kokoro. Other features like whisper are simulated.
        self.emotion_profiles = {
            "happy": {"speed": 1.15, "pitch_mod": 1.05},
            "sad": {"speed": 0.85, "pitch_mod": 0.95},
            "angry": {"speed": 1.25, "pitch_mod": 0.90},
            "whisper": {"speed": 0.90, "pitch_mod": 1.10},
            "neutral": {"speed": 1.0, "pitch_mod": 1.0}
        }

        self._init_model()

    def _init_model(self):
        """Loads the Kokoro model if files are present; otherwise warns and runs mock mode."""
        if Kokoro is None or sf is None:
            print("[EmotionTTS] Required libraries (kokoro-onnx, soundfile) not found. Running in mock mode.")
            return

        if os.path.exists(self.model_path) and os.path.exists(self.voices_path):
            try:
                self.kokoro = Kokoro(self.model_path, self.voices_path)
                print(f"[EmotionTTS] Successfully loaded model from '{self.model_path}'")
            except Exception as e:
                print(f"[EmotionTTS] Failed to initialize Kokoro-ONNX: {e}. Running in mock mode.")
        else:
            print(f"[EmotionTTS] Model or voice files missing (model: '{self.model_path}', voices: '{self.voices_path}'). Running in mock mode.")

    def _parse_emotion(self, text):
        """
        Extracts emotion tags from text (e.g. '[happy] Good morning!').
        Returns the parsed emotion and the cleaned text.
        """
        match = re.search(r'^\[(.*?)\]', text)
        if match:
            emotion = match.group(1).lower()
            clean_text = re.sub(r'^\[.*?\]', '', text).strip()
            if emotion in self.emotion_profiles:
                return emotion, clean_text
        return "neutral", text

    def generate(self, text, output_file=None):
        """
        Generates audio samples for the given text, parsing and applying emotion styling.
        Returns (samples, sample_rate).
        """
        emotion, clean_text = self._parse_emotion(text)
        profile = self.emotion_profiles[emotion]

        if self.kokoro is None:
            print(f"[EmotionTTS] [MOCK] Synthesizing '{emotion.upper()}' style: \"{clean_text}\"")
            # Return 1 second of silence as fallback
            sample_rate = 24000
            samples = np.zeros(sample_rate, dtype=np.float32)
            if output_file and sf:
                sf.write(output_file, samples, sample_rate)
            return samples, sample_rate

        # Generate audio using Kokoro
        try:
            samples, sample_rate = self.kokoro.create(
                clean_text,
                voice=self.default_voice,
                speed=profile['speed'],
                lang="en-us"
            )
        except Exception as e:
            print(f"[EmotionTTS] Kokoro generation failed: {e}. Returning mock silence.")
            sample_rate = 24000
            samples = np.zeros(sample_rate, dtype=np.float32)

        # Apply custom DSP adjustments
        if emotion == "whisper":
            # Whisper simulation: Lower volume and inject subtle white noise
            samples = samples * 0.4
            noise = np.random.normal(0, 0.005, samples.shape)
            samples = samples + noise

        if output_file and sf:
            try:
                sf.write(output_file, samples, sample_rate)
            except Exception as e:
                print(f"[EmotionTTS] Failed to save audio file: {e}")

        return samples, sample_rate

    def play(self, samples, sample_rate):
        """Plays the generated numpy audio array using sounddevice."""
        try:
            import sounddevice as sd
            sd.play(samples, sample_rate)
            sd.wait()
        except ImportError:
            print("[EmotionTTS] sounddevice library not installed. Cannot play audio directly.")
        except Exception as e:
            print(f"[EmotionTTS] Audio playback failed: {e}")
