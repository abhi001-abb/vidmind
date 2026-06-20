import os
import re
import json
import queue
import sqlite3
import threading
import time
import uuid
from contextlib import closing

from flask import Flask, request, jsonify, render_template, Response, stream_with_context

from utils.video_to_audio import extract_audio
from utils.speech_to_text import transcribe_audio
from utils.summarizer import generate_summary
from utils.quiz_generator import generate_quiz
from utils.youtube_downloader import download_youtube_video, YouTubeDownloadError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB limit

UPLOAD_DIR = "uploads"
DB_PATH = os.getenv("JOB_DB_PATH", "jobs.db")
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
FILE_MAX_AGE_SECONDS = int(os.getenv("FILE_MAX_AGE_SECONDS", str(2 * 60 * 60)))  # 2h

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Rate limiting ────────────────────────────────────────────────────────────
# Flask-Limiter isn't guaranteed to be installed everywhere this runs, and the
# whole point of adding a limiter is to stop runaway Gemini/yt-dlp costs even
# on a bare-bones deploy — so we fall back to a tiny in-process limiter rather
# than hard-depending on an extra package.
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address

    limiter = Limiter(get_remote_address, app=app, default_limits=[])
    _HAVE_FLASK_LIMITER = True
except ImportError:
    _HAVE_FLASK_LIMITER = False

    class _NoopLimiter:
        """Minimal fallback: fixed-window per-IP limiter, no external deps."""

        def __init__(self):
            self._hits: dict[str, list[float]] = {}
            self._lock = threading.Lock()

        def limit(self, rule: str):
            # rule format: "N per M minutes" / "N per hour" — we only need the
            # common cases used below.
            count, _, period = rule.partition(" per ")
            count = int(count)
            seconds = {"minute": 60, "hour": 3600, "day": 86400}.get(period.strip(), 60)

            def decorator(fn):
                def wrapped(*args, **kwargs):
                    ip = request.remote_addr or "unknown"
                    now = time.time()
                    with self._lock:
                        hits = [t for t in self._hits.get(ip, []) if now - t < seconds]
                        if len(hits) >= count:
                            return jsonify({"error": "Too many requests. Please wait a moment and try again."}), 429
                        hits.append(now)
                        self._hits[ip] = hits
                    return fn(*args, **kwargs)
                wrapped.__name__ = fn.__name__
                return wrapped
            return decorator

    limiter = _NoopLimiter()


# ─── Job store (SQLite) ───────────────────────────────────────────────────────
# Jobs used to live only in an in-memory dict. That breaks the moment the
# process restarts (deploy, crash) or the app runs with >1 worker process,
# since /stream/<job_id> might land on a worker that never saw /process create
# the job. SQLite gives every worker/process a shared, durable view of job
# state without requiring a separate service like Redis.

_db_lock = threading.Lock()


def _init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,           -- queued | running | done | error
                step TEXT,
                message TEXT,
                summary TEXT,
                quiz TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.commit()


def _db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def job_create(job_id: str):
    now = time.time()
    with _db_lock, closing(_db()) as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, created_at, updated_at) VALUES (?, 'queued', ?, ?)",
            (job_id, now, now),
        )
        conn.commit()


def job_update(job_id: str, **fields):
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k} = ?" for k in fields)
    with _db_lock, closing(_db()) as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE job_id = ?", (*fields.values(), job_id))
        conn.commit()


def job_get(job_id: str):
    with closing(_db()) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


_init_db()

# In-process pub/sub so the SSE endpoint can wake up immediately on updates
# instead of polling the DB in a tight loop. This is per-process, which is
# fine: if /stream lands on a different worker than /process, that worker
# will simply poll the DB (see stream() below) — still correct, just slightly
# less snappy. The DB remains the source of truth either way.
_subscribers: dict[str, list[queue.Queue]] = {}
_subscribers_lock = threading.Lock()


def _notify(job_id: str):
    with _subscribers_lock:
        for q in _subscribers.get(job_id, []):
            q.put(True)


def _subscribe(job_id: str) -> queue.Queue:
    q: queue.Queue = queue.Queue()
    with _subscribers_lock:
        _subscribers.setdefault(job_id, []).append(q)
    return q


def _unsubscribe(job_id: str, q: queue.Queue):
    with _subscribers_lock:
        subs = _subscribers.get(job_id, [])
        if q in subs:
            subs.remove(q)
        if not subs:
            _subscribers.pop(job_id, None)


# ─── Concurrency limiting ─────────────────────────────────────────────────────
# Whisper + moviepy are CPU/RAM heavy. Without a cap, N simultaneous uploads
# can try to load N Whisper models into memory at once and OOM the box.
_job_semaphore = threading.Semaphore(MAX_CONCURRENT_JOBS)


# ─── File cleanup sweep ───────────────────────────────────────────────────────
# cleanup_files() (below) only runs on the happy/error path inside a job. If
# the process is killed (OOM, deploy, SIGKILL) mid-job, that cleanup never
# fires and uploads/ silently fills the disk over time. This sweep is a
# second, independent safety net that runs regardless of how a job died.

def cleanup_files(*paths):
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass


def _sweep_old_uploads():
    while True:
        try:
            cutoff = time.time() - FILE_MAX_AGE_SECONDS
            if os.path.isdir(UPLOAD_DIR):
                for name in os.listdir(UPLOAD_DIR):
                    path = os.path.join(UPLOAD_DIR, name)
                    try:
                        if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                            os.remove(path)
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(600)  # every 10 minutes


threading.Thread(target=_sweep_old_uploads, daemon=True).start()


# ─── YouTube URL validation ───────────────────────────────────────────────────
# youtube_downloader.py is built around yt-dlp, which understands hundreds of
# sites, not just YouTube. Without validating the URL shape here, /process
# would silently accept (and try to download) arbitrary URLs from any site
# yt-dlp supports — a much bigger surface than this app is meant to expose.
_YOUTUBE_URL_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com/(watch\?v=|shorts/|embed/)|youtu\.be/)[\w\-]+",
    re.IGNORECASE,
)


def is_youtube_url(url: str) -> bool:
    return bool(_YOUTUBE_URL_RE.match(url.strip()))


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
@limiter.limit("10 per hour")
def process():
    """
    Accepts a video file upload OR a YouTube URL.
    Creates a durable job row and starts processing in a background thread.
    The browser then opens /stream/<job_id> to receive progress updates.
    """
    video_file = request.files.get("video")
    youtube_url = request.form.get("youtube", "").strip()
    output_language = request.form.get("output_language", "English")

    if not video_file and not youtube_url:
        return jsonify({"error": "No input provided."}), 400
    if video_file and youtube_url:
        return jsonify({"error": "Provide either a video file or a YouTube link, not both."}), 400

    if video_file:
        allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
        ext = os.path.splitext(video_file.filename)[1].lower()
        if ext not in allowed:
            return jsonify({"error": f"Unsupported file type '{ext}'. Upload an mp4/mov/avi/mkv."}), 400

    if youtube_url and not is_youtube_url(youtube_url):
        return jsonify({"error": "That doesn't look like a YouTube URL. Paste a youtube.com or youtu.be link."}), 400

    job_id = str(uuid.uuid4())
    job_create(job_id)

    upload_path = None
    if video_file:
        upload_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
        video_file.save(upload_path)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, upload_path, youtube_url, output_language),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/stream/<job_id>")
def stream(job_id: str):
    """SSE endpoint — browser connects here to receive live progress.

    Reads from the SQLite job row (source of truth) and wakes up either via
    the in-process notify queue (fast path, same worker) or a short poll
    (cross-worker fallback) so this still works under multiple gunicorn
    workers, not just `flask run`.
    """
    row = job_get(job_id)
    if not row:
        return Response(
            "event: error\ndata: {\"message\": \"Unknown or expired job.\"}\n\n",
            mimetype="text/event-stream",
        )

    def generate():
        last_sent = None
        q = _subscribe(job_id)
        try:
            deadline = time.time() + 600  # overall 10-minute cap per stream
            while time.time() < deadline:
                row = job_get(job_id)
                if not row:
                    yield "event: error\ndata: {\"message\": \"Job not found.\"}\n\n"
                    return

                fingerprint = (row["status"], row["step"], row["message"])
                if fingerprint != last_sent:
                    last_sent = fingerprint
                    if row["status"] == "error":
                        yield f"event: error\ndata: {json.dumps({'message': row['error'] or 'Unknown error.'})}\n\n"
                        return
                    if row["status"] == "done":
                        payload = {"summary": row["summary"], "quiz": row["quiz"]}
                        yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                        return
                    if row["step"]:
                        payload = {"step": row["step"], "message": row["message"] or ""}
                        yield f"event: progress\ndata: {json.dumps(payload)}\n\n"

                try:
                    q.get(timeout=2)
                except queue.Empty:
                    pass  # fall through and re-poll the DB

            yield "event: error\ndata: {\"message\": \"Timed out waiting for job to finish.\"}\n\n"
        finally:
            _unsubscribe(job_id, q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Pipeline (runs in background thread) ────────────────────────────────────

def _run_pipeline(job_id: str, upload_path, youtube_url: str, output_language: str):
    video_path = None
    audio_path = None

    def progress(step: str, message: str):
        job_update(job_id, status="running", step=step, message=message)
        _notify(job_id)

    def fail(message: str):
        job_update(job_id, status="error", error=message)
        _notify(job_id)

    # Cap how many pipelines run at once — Whisper + moviepy are heavy enough
    # that unlimited concurrency can OOM the process under modest load.
    acquired = _job_semaphore.acquire(timeout=1)
    if not acquired:
        progress("queued", "Server is busy — waiting for a free slot…")
        _job_semaphore.acquire()  # block until a slot frees up

    try:
        # ── Step 1: acquire video ──────────────────────────────────────────
        if youtube_url:
            progress("download", "⬇️  Downloading YouTube video…")
            try:
                video_path = download_youtube_video(youtube_url)
            except YouTubeDownloadError as e:
                fail(str(e))
                return
            if not video_path:
                fail("YouTube download failed for an unknown reason. Try uploading the video file directly.")
                return
        else:
            video_path = upload_path

        # ── Step 2: extract audio ──────────────────────────────────────────
        progress("audio", "🔊 Extracting audio from video…")
        audio_path = extract_audio(video_path)
        if not audio_path:
            fail("Could not extract audio. Make sure the video has an audio track.")
            return

        # ── Step 3: transcribe ─────────────────────────────────────────────
        progress("transcribe", "🧠 Transcribing speech (this may take a minute)…")
        text = transcribe_audio(audio_path)
        if not text:
            fail("Transcription failed or the audio contained no speech.")
            return

        # ── Step 4: summarize ──────────────────────────────────────────────
        progress("summarize", "📝 Generating summary…")
        summary = generate_summary(text, output_language)
        if not summary:
            fail("Summary generation failed.")
            return

        # ── Step 5: quiz ───────────────────────────────────────────────────
        progress("quiz", "❓ Generating quiz questions…")
        quiz = generate_quiz(summary, output_language)
        if not quiz:
            fail("Quiz generation failed.")
            return

        job_update(job_id, status="done", summary=summary, quiz=quiz)
        _notify(job_id)

    except Exception as e:
        fail(f"Unexpected error: {str(e)}")

    finally:
        _job_semaphore.release()
        cleanup_files(audio_path)
        if youtube_url and video_path:
            cleanup_files(video_path)
        elif upload_path:
            cleanup_files(upload_path)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
