"""
youtube_downloader.py
─────────────────────
Downloads a YouTube video to a temp file using yt-dlp.

Keep yt-dlp updated to avoid YouTube bot-detection breaks:
    pip install -U yt-dlp
"""

import os
import uuid
import yt_dlp
from typing import Optional

def download_youtube_video(url: str) -> Optional[str]:
    """
    Download a YouTube video to uploads/<uuid>.mp4.

    Returns the file path on success, None on failure.
    Caller is responsible for deleting the file after use.
    """
    if not url or not url.strip():
        print("❌ Empty URL provided.")
        return None

    os.makedirs("uploads", exist_ok=True)
    filename = str(uuid.uuid4()) + ".mp4"
    output_path = os.path.join("uploads", filename)

    ydl_opts = {
        # Prefer best single-file mp4; fall back to any format and remux to mp4
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # Rotate through common browser User-Agents to reduce bot blocks
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
        # Retry up to 3 times on network errors
        "retries": 3,
        "fragment_retries": 3,
        # Abort if the video is longer than 2 hours (avoids huge downloads)
        "match_filter": yt_dlp.utils.match_filter_func("duration < 7200"),
        # Postprocessor: ensure output is a proper mp4
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                print("❌ yt-dlp returned no info for:", url)
                return None

        # yt-dlp sometimes appends the format id to the filename
        if not os.path.exists(output_path):
            # Try to find what it actually saved
            base = os.path.splitext(output_path)[0]
            for f in os.listdir("uploads"):
                if f.startswith(os.path.basename(base)):
                    output_path = os.path.join("uploads", f)
                    break

        if not os.path.exists(output_path):
            print("❌ Download appeared to succeed but output file not found.")
            return None

        print(f"✅ YouTube video saved: {output_path}")
        return output_path

    except yt_dlp.utils.DownloadError as e:
        print(f"❌ yt-dlp DownloadError: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected download error: {e}")
        return None