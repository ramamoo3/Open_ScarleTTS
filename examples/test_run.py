"""Basic smoke test for Open_ScarleTTS.

Runs in mock mode automatically when model files are absent.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from open_scarletts import EmotionTTS


def main():
    print("--- Open_ScarleTTS Test Run ---")

    with EmotionTTS() as tts:
        test_sentences = [
            "[neutral] System initialized and ready.",
            "[happy] I am so excited to explore the universe with you!",
            "[sad] Oh no... the hyperdrive is completely broken.",
            "[angry] Intruder alert! Step away from the console!",
            "[whisper] It's very quiet in here... don't wake the alien.",
        ]

        try:
            import sounddevice  # noqa: F401

            can_play = True
        except ImportError:
            can_play = False

        for idx, sentence in enumerate(test_sentences):
            filename = f"test_emotion_{idx}.wav"
            print(f"\nGenerating audio for: {sentence}")
            samples, rate = tts.generate(sentence, output_file=filename)

            if can_play:
                print("Playing audio through sounddevice...")
                tts.play(samples, rate)
            else:
                print(f"Audio saved to '{filename}'. Install 'sounddevice' to play it live.")


if __name__ == "__main__":
    main()
