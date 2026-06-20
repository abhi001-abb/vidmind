"""
summarizer.py
─────────────
Summarizes transcribed text.

Two backends are available, selected via the SUMMARIZER_BACKEND env var:

  - "gemini" (default): calls Google's Gemini API. Supports any output
    language, but requires GOOGLE_API_KEY and costs money per call.
  - "bart": runs facebook/bart-large-cnn locally via transformers. Free and
    works offline, but BART-large-cnn is English-only — if a non-English
    output_language is requested with this backend, it's ignored.

This used to be ~150 lines of working BART code commented out underneath the
Gemini implementation. That's dead weight in the file (confusing, easy to
forget about, and never executes), so it's been restored as a real, working
fallback instead of inert comments.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re

BACKEND = os.getenv("SUMMARIZER_BACKEND", "gemini").strip().lower()


# ─── Gemini backend ────────────────────────────────────────────────────────────

def _generate_summary_gemini(text: str, output_language: str) -> str | None:
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ generate_summary: GOOGLE_API_KEY is not set.")
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are an expert educational summarizer.

Create a concise and well-structured bullet-point summary.

Requirements:
- Generate the summary in {output_language}.
- Keep the meaning accurate.
- Use bullet points.
- Focus on key concepts and important details.
- Do not add information that is not present in the transcript.

Transcript:
{text}
"""

    try:
        print(f"🌐 Summary Language: {output_language}")
        print("📤 Sending summary request to Gemini...")

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1500,
            },
        )

        print("📥 Summary received from Gemini")
        return response.text.strip()

    except Exception as e:
        print(f"❌ generate_summary (gemini) error: {e}")
        return None


# ─── BART backend (local, offline, English-only) ───────────────────────────────

MODEL_NAME = os.getenv("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
MAX_INPUT_TOKENS = 1024     # BART's hard limit
MAX_OUTPUT_TOKENS = 256     # output per chunk
MIN_OUTPUT_TOKENS = 80

_bart_summarizer = None
_bart_tokenizer = None


def _ensure_nltk():
    import nltk
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def _get_bart_pipeline():
    global _bart_summarizer, _bart_tokenizer
    if _bart_summarizer is None:
        from transformers import pipeline, AutoTokenizer
        print(f"🚀 Loading summarizer model '{MODEL_NAME}'… (first request only)")
        _bart_summarizer = pipeline("summarization", model=MODEL_NAME)
        _bart_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("✅ Summarizer ready.")
    return _bart_summarizer, _bart_tokenizer


def _clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)           # remove [Music], [Applause] etc.
    text = re.sub(r"\(.*?\)", "", text)           # remove (inaudible) etc.
    text = re.sub(r"https?://\S+", "", text)      # remove URLs
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_by_tokens(text: str, tokenizer, max_tokens: int = MAX_INPUT_TOKENS - 10) -> list[str]:
    """Split *text* into chunks that each fit within *max_tokens* tokens.
    Splits on sentence boundaries when possible for better coherence."""
    from nltk.tokenize import sent_tokenize

    _ensure_nltk()
    sentences = sent_tokenize(text)
    chunks = []
    current_sents = []
    current_len = 0

    for sent in sentences:
        sent_len = len(tokenizer.encode(sent, add_special_tokens=False))

        if sent_len >= max_tokens:
            if current_sents:
                chunks.append(" ".join(current_sents))
                current_sents, current_len = [], 0
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i: i + max_tokens]
                chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
            continue

        if current_len + sent_len > max_tokens:
            if current_sents:
                chunks.append(" ".join(current_sents))
            current_sents = [sent]
            current_len = sent_len
        else:
            current_sents.append(sent)
            current_len += sent_len

    if current_sents:
        chunks.append(" ".join(current_sents))

    return [c for c in chunks if c.strip()]


def _summarize_chunk(chunk: str, summarizer) -> str | None:
    try:
        input_len = len(chunk.split())
        max_len = min(MAX_OUTPUT_TOKENS, max(MIN_OUTPUT_TOKENS, int(input_len * 0.6)))
        min_len = min(MIN_OUTPUT_TOKENS, max_len - 10)

        result = summarizer(
            chunk,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"].strip()
    except Exception as e:
        print(f"⚠️  Chunk summarization failed: {e}")
        return None


def _remove_redundancy(sentences: list[str]) -> list[str]:
    """Remove sentences that are near-duplicates of an earlier one."""
    unique = []
    for s in sentences:
        s_lower = s.lower()
        if not any(s_lower in u.lower() or u.lower() in s_lower for u in unique):
            unique.append(s)
    return unique


def _format_summary(raw: str) -> str:
    from nltk.tokenize import sent_tokenize

    _ensure_nltk()
    sentences = sent_tokenize(raw)
    sentences = list(dict.fromkeys(sentences))       # exact dedup
    sentences = _remove_redundancy(sentences)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    max_lines = min(20, max(8, len(sentences)))
    return "\n".join(f"• {s}" for s in sentences[:max_lines])


def _generate_summary_bart(text: str, output_language: str) -> str | None:
    if output_language.strip().lower() != "english":
        print(f"⚠️  BART backend is English-only; ignoring requested language '{output_language}'.")

    try:
        summarizer, tokenizer = _get_bart_pipeline()
        text = _clean_text(text)

        if len(text.split()) < 30:
            print("⚠️  Text too short to summarize — returning as-is.")
            return text

        chunks = _chunk_by_tokens(text, tokenizer)
        print(f"📄 Summarizing {len(chunks)} chunk(s)…")

        summaries = []
        for i, chunk in enumerate(chunks):
            s = _summarize_chunk(chunk, summarizer)
            if s:
                summaries.append(s)
            print(f"   Chunk {i + 1}/{len(chunks)} done.")

        if not summaries:
            print("❌ All chunks failed to summarize.")
            return None

        combined = " ".join(summaries)

        if len(summaries) > 1 and len(combined.split()) > MAX_OUTPUT_TOKENS:
            print("📄 Running second-pass summary for coherence…")
            second_chunks = _chunk_by_tokens(combined, tokenizer)
            second_summaries = [_summarize_chunk(c, summarizer) for c in second_chunks]
            second_summaries = [s for s in second_summaries if s]
            if second_summaries:
                combined = " ".join(second_summaries)

        result = _format_summary(combined)
        print("✅ Summary complete.")
        return result

    except Exception as e:
        print(f"❌ generate_summary (bart) error: {e}")
        return None


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_summary(text: str, output_language: str = "English") -> str | None:
    """Summarize *text* and return a bullet-point string, or None on failure.

    Backend is chosen by the SUMMARIZER_BACKEND env var ("gemini" or "bart").
    """
    if not text or not text.strip():
        print("❌ generate_summary: empty input.")
        return None

    if BACKEND == "bart":
        return _generate_summary_bart(text, output_language)
    return _generate_summary_gemini(text, output_language)
