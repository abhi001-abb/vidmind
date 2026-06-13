from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
import google.generativeai as genai

_model = None


def _get_model():
    global _model
    if _model is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel("gemini-2.5-flash")
    return _model


def _build_system_prompt(language: str = "English") -> str:
    return f"""
You are an expert educational quiz creator.

Given a text summary, generate multiple-choice questions that test genuine comprehension.

Rules:
- Each question must be answerable from the summary text only.
- Distractors must be plausible but incorrect.
- One and only one correct answer.
- Cover different aspects of the content.
- Keep questions concise.
- Generate ALL questions, options, and explanations in {language}.
- Return ONLY valid JSON. No markdown. No explanation outside JSON.

Format:
[
  {{
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A",
    "explanation": "Brief reason why this is correct."
  }}
]
"""


def _try_repair_json(raw: str) -> list | None:
    """Try to salvage truncated JSON by extracting complete question objects."""
    try:
        # Find all complete question blocks using regex
        pattern = r'\{\s*"question".*?"explanation"\s*:\s*"[^"]*"\s*\}'
        matches = re.findall(pattern, raw, re.DOTALL)
        if not matches:
            return None
        repaired = "[" + ",".join(matches) + "]"
        return json.loads(repaired)
    except Exception:
        return None


def generate_quiz(
    summary: str,
    output_language: str = "English",
    num_questions: int = 5
) -> str | None:

    if not summary or not summary.strip():
        print("❌ generate_quiz: empty summary.")
        return None

    clean_summary = re.sub(r"^•\s*", "", summary, flags=re.MULTILINE).strip()

    if len(clean_summary.split()) < 20:
        print("⚠️ Summary too short.")
        return None

    clean_summary = clean_summary[:4000]

    prompt = f"""
{_build_system_prompt(output_language)}

Generate exactly {num_questions} MCQs in {output_language}.

SUMMARY:

{clean_summary}
"""

    try:
        model = _get_model()

        print(f"🤖 Asking Gemini to generate {num_questions} quiz questions...")
        print(f"🌐 Quiz Language: {output_language}")
        print(f"📊 Summary length: {len(clean_summary)} chars / {len(clean_summary.split())} words")
        print("📤 Sending request to Gemini...")

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 2500,  # ← increased from 1200
            }
        )

        print("📥 Received response from Gemini")

        raw = response.text.strip()
        print(f"📊 Response length: {len(raw)} chars")

        # Remove markdown fences
        raw = re.sub(r"^```json\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"^```\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

        # Try normal parse first
        try:
            questions = json.loads(raw)
        except json.JSONDecodeError:
            print("⚠️ JSON truncated or malformed — attempting repair...")
            questions = _try_repair_json(raw)
            if questions is None:
                print("❌ JSON repair failed.")
                print("Raw response:")
                print(raw)
                return None
            print(f"🔧 Repaired JSON — recovered {len(questions)} questions")

        if not isinstance(questions, list) or len(questions) == 0:
            print("❌ No valid questions returned.")
            return None

        print(f"✅ Generated {len(questions)} questions")
        return _format_quiz(questions)

    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return None


def _format_quiz(questions):
    lines = []
    for i, q in enumerate(questions, 1):
        lines.append(f"Q{i}: {q.get('question', '')}")
        for opt in q.get("options", []):
            lines.append(f"   {opt}")
        lines.append(
            f"   ✅ Answer: {q.get('answer', '')}"
            f" — {q.get('explanation', '')}"
        )
        lines.append("")
    return "\n".join(lines).strip()