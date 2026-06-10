import os
import sys

# Append parent directory to path to enable package import without installation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from open_scarletts import EmotionTTS

def main():
    print("--- Open_ScarleTTS Test Run ---")
    
    # Initialize wrapper.
    # It will automatically fall back to mock mode if model files are not in the current path.
    tts = EmotionTTS()

    test_sentences = [
        "[neutral] System initialized and ready.",
        "[happy] I am so excited to explore the universe with you!",
        "[sad] Oh no... the hyperdrive is completely broken.",
        "[angry] Intruder alert! Step away from the console!",
        "[whisper] It's very quiet in here... don't wake the alien."
    ]

    for idx, sentence in enumerate(test_sentences):
        filename = f"test_emotion_{idx}.wav"
        print(f"\nGenerating audio for: {sentence}")
        samples, rate = tts.generate(sentence, output_file=filename)
        
        try:
            import sounddevice
            print(f"Playing audio through sounddevice...")
            tts.play(samples, rate)
        except ImportError:
            print(f"Audio saved to '{filename}'. (Install 'sounddevice' to play audio in real time).")

if __name__ == "__main__":
    main()
