"""
youtube_downloader.py
─────────────────────
Downloads a YouTube video to a temp file using yt-dlp.

Keep yt-dlp updated to avoid YouTube bot-detection breaks:
    pip install -U yt-dlp

Note on reliability: YouTube actively fights automated downloads, and cloud
provider IP ranges (AWS/GCP/Azure/etc.) get flagged harder than residential
IPs. No amount of code here makes this 100% reliable — that's an ongoing
arms race between yt-dlp and YouTube, not a bug in this file. The two real
levers available:
  1. Keep yt-dlp itself up to date (it ships fixes for new blocks quickly).
  2. Set YTDLP_COOKIES_FILE to a cookies.txt exported from a logged-in
     browser session — this is the standard yt-dlp workaround for
     bot-detection / age-gated / region-locked content.
"""

import os
import uuid
import yt_dlp
from typing import Optional


class YouTubeDownloadError(Exception):
    """Raised with a human-readable reason so the caller can show it
    directly instead of guessing why the download failed."""


def download_youtube_video(url: str) -> Optional[str]:
    """
    Download a YouTube video to uploads/<uuid>.mp4.

    Returns the file path on success.
    Raises YouTubeDownloadError with a specific, user-facing reason on
    failure (previously this just returned None, forcing the caller to show
    one generic guess — "private, age-restricted, or rate-limited" — for
    every possible failure, even ones that were actually something else
    entirely, like an unsupported URL or a network timeout).

    Caller is responsible for deleting the file after use.
    """
    if not url or not url.strip():
        raise YouTubeDownloadError("No URL provided.")

    os.makedirs("uploads", exist_ok=True)
    filename = str(uuid.uuid4()) + ".mp4"
    output_path = os.path.join("uploads", filename)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
        "retries": 3,
        "fragment_retries": 3,
        "match_filter": yt_dlp.utils.match_filter_func("duration < 7200"),
        "postprocessors": [
            {"key": "FFmpegVideoConvertor", "preferedformat": "mp4"},
        ],
    }

    cookies_file = os.getenv("YTDLP_COOKIES_FILE")
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise YouTubeDownloadError("YouTube returned no video info for that link.")

        if not os.path.exists(output_path):
            base = os.path.splitext(output_path)[0]
            for f in os.listdir("uploads"):
                if f.startswith(os.path.basename(base)):
                    output_path = os.path.join("uploads", f)
                    break

        if not os.path.exists(output_path):
            raise YouTubeDownloadError("Download appeared to succeed but the output file wasn't found.")

        print(f"✅ YouTube video saved: {output_path}")
        return output_path

    except yt_dlp.utils.DownloadError as e:
        reason = _classify_ytdlp_error(str(e))
        print(f"❌ yt-dlp DownloadError: {e}")
        raise YouTubeDownloadError(reason) from e
    except YouTubeDownloadError:
        raise
    except Exception as e:
        print(f"❌ Unexpected download error: {e}")
        raise YouTubeDownloadError(f"Unexpected error while downloading: {e}") from e


def _classify_ytdlp_error(raw_message: str) -> str:
    """Turn yt-dlp's raw error text into a short, specific, user-facing reason."""
    msg = raw_message.lower()

    if "sign in to confirm" in msg or "bot" in msg:
        return "YouTube is blocking this download as suspected bot traffic. Try uploading the video file directly instead."
    if "private video" in msg:
        return "This video is private and can't be downloaded."
    if "age" in msg and "restrict" in msg:
        return "This video is age-restricted and can't be downloaded without an authenticated session."
    if "unavailable" in msg:
        return "This video is unavailable (it may have been removed or region-blocked)."
    if "duration" in msg:
        return "This video is longer than the 2-hour limit."
    if "unsupported url" in msg:
        return "That URL isn't a supported YouTube video link."
    if "timed out" in msg or "timeout" in msg:
        return "The download timed out. Please try again."

    return "Could not download this video. Try uploading the video file directly instead."
