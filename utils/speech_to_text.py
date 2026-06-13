"""
speech_to_text.py
─────────────────
Transcribes audio using OpenAI Whisper.

Model is lazy-loaded on first use so Flask starts instantly.
Uses the "small" model by default — significantly more accurate than "base"
(~88% vs ~74% word accuracy on English) with only a modest speed cost.

Swap to "medium" for even better accuracy on noisy/accented audio,
or "large-v3" if you have a GPU and need maximum accuracy.
"""

import os
import whisper

# Module-level cache — loaded once, reused for all requests
_model = None
_model_name = os.getenv("WHISPER_MODEL", "small")   # override via env var
from typing import Optional

def _get_model():
    global _model
    if _model is None:
        print(f"🚀 Loading Whisper '{_model_name}' model… (first request only)")
        _model = whisper.load_model(_model_name)
        print("✅ Whisper model ready.")
    return _model


def transcribe_audio(audio_path: str) -> Optional[str]:
    """
    Transcribe *audio_path* and return the text, or None on failure.

    Whisper options used:
    - fp16=False      : required for CPU inference
    - language=None   : auto-detect language (set to "en" to force English)
    - beam_size=5     : better accuracy than greedy (default beam_size=1)
    - best_of=5       : sample multiple candidates at temperature>0
    - condition_on_previous_text=False : prevents hallucination loops on
                         silent/noisy segments
    - no_speech_threshold=0.6          : skip segments that are likely silence
    """
    if not audio_path:
        print("❌ transcribe_audio: no audio path provided.")
        return None

    if not os.path.exists(audio_path):
        print(f"❌ transcribe_audio: file not found — {audio_path}")
        return None

    try:
        model = _get_model()
        print("🧠 Transcribing audio…")

        result = model.transcribe(
            audio_path,
            fp16=False,
            beam_size=5,
            best_of=5,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            verbose=False,
        )

        text = result.get("text", "").strip()

        if not text:
            print("⚠️  Transcription returned empty text — audio may be silent or inaudible.")
            return None

        print(f"✅ Transcription complete ({len(text.split())} words).")
        return text

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return None