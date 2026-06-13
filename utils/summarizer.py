# """
# summarizer.py
# ─────────────
# Summarizes transcribed text using facebook/bart-large-cnn.

# Key fixes vs original:
# - Uses the BART tokenizer to chunk by actual tokens (not word count),
#   preventing silent truncation at the 1024-token limit.
# - Lazy-loads the pipeline so Flask starts instantly.
# - Better chunk overlap so context isn't lost at boundaries.
# - Improved redundancy removal.
# - Graceful null returns on every failure path.
# """
from dotenv import load_dotenv
load_dotenv()

import os
import google.generativeai as genai

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

def generate_summary(text, output_language="English"):
    try:

        print(f"🌐 Summary Language: {output_language}")

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

        print("📤 Sending summary request to Gemini...")

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1500
            }
        )

        print("📥 Summary received from Gemini")

        return response.text.strip()

    except Exception as e:
        print(f"❌ generate_summary error: {e}")
        return None





# import re
# import os
# from functools import lru_cache

# import nltk
# from nltk.tokenize import sent_tokenize
# from transformers import pipeline, AutoTokenizer

# # ─── Lazy setup ──────────────────────────────────────────────────────────────

# MODEL_NAME = os.getenv("SUMMARIZER_MODEL", "facebook/bart-large-cnn")
# MAX_INPUT_TOKENS = 1024     # BART's hard limit
# MAX_OUTPUT_TOKENS = 256     # output per chunk
# MIN_OUTPUT_TOKENS = 80

# _summarizer = None
# _tokenizer = None


# def _ensure_nltk():
#     for resource in ("punkt", "punkt_tab"):
#         try:
#             nltk.data.find(f"tokenizers/{resource}")
#         except LookupError:
#             nltk.download(resource, quiet=True)


# def _get_pipeline():
#     global _summarizer, _tokenizer
#     if _summarizer is None:
#         print(f"🚀 Loading summarizer model '{MODEL_NAME}'… (first request only)")
#         _summarizer = pipeline("summarization", model=MODEL_NAME)
#         _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         print("✅ Summarizer ready.")
#     return _summarizer, _tokenizer


# # ─── Text helpers ─────────────────────────────────────────────────────────────

# def _clean_text(text: str) -> str:
#     text = re.sub(r"\[.*?\]", "", text)           # remove [Music], [Applause] etc.
#     text = re.sub(r"\(.*?\)", "", text)           # remove (inaudible) etc.
#     text = re.sub(r"https?://\S+", "", text)      # remove URLs
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# def _chunk_by_tokens(text: str, tokenizer, max_tokens: int = MAX_INPUT_TOKENS - 10) -> list[str]:
#     """
#     Split *text* into chunks that each fit within *max_tokens* tokens.
#     Splits on sentence boundaries when possible for better coherence.
#     """
#     _ensure_nltk()
#     sentences = sent_tokenize(text)
#     chunks = []
#     current_sents = []
#     current_len = 0

#     for sent in sentences:
#         sent_len = len(tokenizer.encode(sent, add_special_tokens=False))

#         # Single sentence longer than limit — hard-split it
#         if sent_len >= max_tokens:
#             if current_sents:
#                 chunks.append(" ".join(current_sents))
#                 current_sents, current_len = [], 0
#             # Tokenize and split by raw tokens
#             tokens = tokenizer.encode(sent, add_special_tokens=False)
#             for i in range(0, len(tokens), max_tokens):
#                 chunk_tokens = tokens[i : i + max_tokens]
#                 chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
#             continue

#         if current_len + sent_len > max_tokens:
#             if current_sents:
#                 chunks.append(" ".join(current_sents))
#             current_sents = [sent]
#             current_len = sent_len
#         else:
#             current_sents.append(sent)
#             current_len += sent_len

#     if current_sents:
#         chunks.append(" ".join(current_sents))

#     return [c for c in chunks if c.strip()]


# def _summarize_chunk(chunk: str, summarizer) -> str | None:
#     try:
#         input_len = len(chunk.split())
#         max_len = min(MAX_OUTPUT_TOKENS, max(MIN_OUTPUT_TOKENS, int(input_len * 0.6)))
#         min_len = min(MIN_OUTPUT_TOKENS, max_len - 10)

#         result = summarizer(
#             chunk,
#             max_length=max_len,
#             min_length=min_len,
#             do_sample=False,
#             truncation=True,
#         )
#         return result[0]["summary_text"].strip()
#     except Exception as e:
#         print(f"⚠️  Chunk summarization failed: {e}")
#         return None


# def _remove_redundancy(sentences: list[str]) -> list[str]:
#     """Remove sentences that are near-duplicates of an earlier one."""
#     unique = []
#     for s in sentences:
#         s_lower = s.lower()
#         if not any(
#             s_lower in u.lower() or u.lower() in s_lower
#             for u in unique
#         ):
#             unique.append(s)
#     return unique


# def _format_summary(raw: str) -> str:
#     _ensure_nltk()
#     sentences = sent_tokenize(raw)
#     sentences = list(dict.fromkeys(sentences))       # exact dedup
#     sentences = _remove_redundancy(sentences)
#     sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
#     max_lines = min(20, max(8, len(sentences)))
#     return "\n".join(f"• {s}" for s in sentences[:max_lines])


# # ─── Public API ───────────────────────────────────────────────────────────────

# def generate_summary(text: str) -> str | None:
#     """
#     Summarize *text* and return a bullet-point string, or None on failure.
#     """
#     if not text or not text.strip():
#         print("❌ generate_summary: empty input.")
#         return None

#     try:
#         summarizer, tokenizer = _get_pipeline()
#         text = _clean_text(text)

#         if len(text.split()) < 30:
#             print("⚠️  Text too short to summarize — returning as-is.")
#             return text

#         chunks = _chunk_by_tokens(text, tokenizer)
#         print(f"📄 Summarizing {len(chunks)} chunk(s)…")

#         summaries = []
#         for i, chunk in enumerate(chunks):
#             s = _summarize_chunk(chunk, summarizer)
#             if s:
#                 summaries.append(s)
#             print(f"   Chunk {i+1}/{len(chunks)} done.")

#         if not summaries:
#             print("❌ All chunks failed to summarize.")
#             return None

#         combined = " ".join(summaries)

#         # If we had multiple chunks, do a second-pass summary for coherence
#         if len(summaries) > 1 and len(combined.split()) > MAX_OUTPUT_TOKENS:
#             print("📄 Running second-pass summary for coherence…")
#             second_chunks = _chunk_by_tokens(combined, tokenizer)
#             second_summaries = [_summarize_chunk(c, summarizer) for c in second_chunks]
#             second_summaries = [s for s in second_summaries if s]
#             if second_summaries:
#                 combined = " ".join(second_summaries)

#         result = _format_summary(combined)
#         print("✅ Summary complete.")
#         return result

#     except Exception as e:
#         print(f"❌ generate_summary error: {e}")
#         return None