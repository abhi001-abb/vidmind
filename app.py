import os
import json
import queue
import threading
import uuid
from flask import Flask, request, jsonify, render_template, Response, stream_with_context

from utils.video_to_audio import extract_audio
from utils.speech_to_text import transcribe_audio
from utils.summarizer import generate_summary
from utils.quiz_generator import generate_quiz
from utils.youtube_downloader import download_youtube_video

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB limit

# ─── Progress via Server-Sent Events ─────────────────────────────────────────
# Each job gets a queue; the processing thread pushes steps, the SSE endpoint
# streams them to the browser in real time.

_job_queues: dict[str, queue.Queue] = {}


def push(q: queue.Queue, event: str, data: dict):
    q.put({"event": event, "data": data})


def cleanup_files(*paths):
    """Safely delete temp files after processing."""
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Accepts a video file upload OR a YouTube URL.
    Starts processing in a background thread and returns a job_id immediately.
    The browser then opens /stream/<job_id> to receive SSE progress updates.
    """
    video_file = request.files.get("video")
    youtube_url = request.form.get("youtube", "").strip()
    output_language = request.form.get(
    "output_language",
    "English"
    )
    if not video_file and not youtube_url:
        return jsonify({"error": "No input provided."}), 400
    if video_file and youtube_url:
        return jsonify({"error": "Provide either a video file or a YouTube link, not both."}), 400

    # Validate file type if upload
    if video_file:
        allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
        ext = os.path.splitext(video_file.filename)[1].lower()
        if ext not in allowed:
            return jsonify({"error": f"Unsupported file type '{ext}'. Upload an mp4/mov/avi/mkv."}), 400

    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()
    _job_queues[job_id] = q

    # Save upload to a temp path now (before the thread starts) so the file
    # object isn't garbage-collected.
    upload_path = None
    if video_file:
        os.makedirs("uploads", exist_ok=True)
        upload_path = os.path.join("uploads", f"{job_id}{ext}")
        video_file.save(upload_path)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, q, upload_path, youtube_url,output_language),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    """SSE endpoint — browser connects here to receive live progress."""
    q = _job_queues.get(job_id)
    if not q:
        return Response("data: {\"error\": \"Unknown job\"}\n\n", mimetype="text/event-stream")

    def generate():
        while True:
            try:
                msg = q.get(timeout=300)          # wait up to 2 min per step
            except queue.Empty:
                yield "data: {\"error\": \"Timeout\"}\n\n"
                break

            payload = json.dumps(msg["data"])
            yield f"event: {msg['event']}\ndata: {payload}\n\n"

            if msg["event"] in ("done", "error"):
                _job_queues.pop(job_id, None)
                break

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",        # disable Nginx buffering if present
        },
    )


# ─── Pipeline (runs in background thread) ────────────────────────────────────

def _run_pipeline(job_id: str,q: queue.Queue,upload_path: str | None, youtube_url: str,output_language: str):
    video_path = None
    audio_path = None

    try:
        # ── Step 1: acquire video ──────────────────────────────────────────
        if youtube_url:
            push(q, "progress", {"step": "download", "message": "⬇️  Downloading YouTube video…"})
            video_path = download_youtube_video(youtube_url)
            if not video_path:
                push(q, "error", {"message": "YouTube download failed. The link may be private, age-restricted, or rate-limited. Try uploading the video file directly."})
                return
        else:
            video_path = upload_path

        # ── Step 2: extract audio ──────────────────────────────────────────
        push(q, "progress", {"step": "audio", "message": "🔊 Extracting audio from video…"})
        audio_path = extract_audio(video_path)
        if not audio_path:
            push(q, "error", {"message": "Could not extract audio. Make sure the video has an audio track."})
            return

        # ── Step 3: transcribe ─────────────────────────────────────────────
        push(q, "progress", {"step": "transcribe", "message": "🧠 Transcribing speech (this may take a minute)…"})
        text = transcribe_audio(audio_path)
        if not text:
            push(q, "error", {"message": "Transcription failed or the audio contained no speech."})
            return

        # ── Step 4: summarize ──────────────────────────────────────────────
        push(q, "progress", {"step": "summarize", "message": "📝 Generating summary…"})
        summary = generate_summary(text,output_language)
        if not summary:
            push(q, "error", {"message": "Summary generation failed."})
            return

        # ── Step 5: quiz ───────────────────────────────────────────────────
        push(q, "progress", {"step": "quiz", "message": "❓ Generating quiz questions…"})
        quiz = generate_quiz(summary,output_language)
        if not quiz:
            push(q, "error", {"message": "Quiz generation failed."})
            return

        push(q, "done", {"summary": summary, "quiz": quiz})

    except Exception as e:
        push(q, "error", {"message": f"Unexpected error: {str(e)}"})

    finally:
        # Always clean up temp files
        cleanup_files(audio_path)
        if youtube_url and video_path:   # only delete YT downloads, not user uploads
            cleanup_files(video_path)
        elif upload_path:
            cleanup_files(upload_path)


if __name__ == "__main__":
    app.run(debug=True, port=5001)