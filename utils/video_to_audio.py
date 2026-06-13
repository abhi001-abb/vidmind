"""
video_to_audio.py
─────────────────
Extracts the audio track from a video file and saves it as a 16kHz mono WAV.

16kHz mono is the exact sample rate Whisper expects — feeding it a different
rate causes silent resampling that degrades transcription accuracy.
"""

import os
from moviepy.editor import VideoFileClip


from typing import Optional

def extract_audio(video_path: str) -> Optional[str]:
    """
    Extract audio from *video_path* and write a 16kHz mono WAV next to it.

    Returns the WAV path on success, None on failure.
    """
    if not video_path:
        print("❌ extract_audio: no video path provided.")
        return None

    if not os.path.exists(video_path):
        print(f"❌ extract_audio: file not found — {video_path}")
        return None

    base, _ = os.path.splitext(video_path)
    audio_path = base + ".wav"

    video = None
    try:
        print(f"🎬 Opening video: {video_path}")
        video = VideoFileClip(video_path)

        if video.audio is None:
            print("❌ No audio track found in the video.")
            return None

        print("🔊 Extracting audio (16kHz mono)…")
        video.audio.write_audiofile(
            audio_path,
            fps=16000,          # Whisper's native sample rate
            nbytes=2,           # 16-bit PCM
            ffmpeg_params=["-ac", "1"],   # mono
            logger=None,        # suppress MoviePy progress bar
        )

        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            print("❌ Audio file was not created or is empty.")
            return None

        print(f"✅ Audio saved: {audio_path}")
        return audio_path

    except Exception as e:
        print(f"❌ Audio extraction failed: {e}")
        return None

    finally:
        if video is not None:
            try:
                video.close()
            except Exception:
                pass