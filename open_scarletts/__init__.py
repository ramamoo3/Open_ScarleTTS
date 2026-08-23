"""Open_ScarleTTS: lightweight emotion-styled TTS for edge devices."""

from .cache import PhraseCache, make_key
from .speaker import play_stream, tee_to_file
from .textsplit import split_sentences
from .tts import EmotionTTS

__version__ = "0.3.0"

__all__ = [
    "EmotionTTS",
    "PhraseCache",
    "make_key",
    "play_stream",
    "tee_to_file",
    "split_sentences",
    "__version__",
]
