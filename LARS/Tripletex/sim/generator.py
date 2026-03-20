"""Task prompt generator using Gemini.

Generates realistic task prompts + expected field values for simulation.
"""

import json
import random

from google import genai

from config import GOOGLE_API_KEY, GEMINI_MODEL
from sim.task_definitions import TaskDef, LANGUAGES


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


def generate_task(task_def: TaskDef, language: str = "") -> dict:
    """Generate a task prompt and expected values using Gemini.

    Args:
        task_def: The task definition to generate for.
        language: Language code (e.g. "no", "en"). Random if empty.

    Returns:
        {
            "prompt": str,           # The task prompt text
            "language": str,         # Language code used
            "expected": dict,        # Expected field values
            "task_def": TaskDef,     # Reference to the task definition
        }
    """
    if not language:
        language = random.choice(list(LANGUAGES.keys()))

    lang_name = LANGUAGES[language]

    system_prompt = f"""\
You are a task generator for a Tripletex accounting competition simulator.
Generate a realistic accounting task prompt and the expected field values.

RULES:
- Write the prompt in {lang_name}
- Use realistic Scandinavian/European names and company names
- Dates should be in March 2026
- Emails should use domains like example.com, test.no, or firma.no
- Phone numbers in Norwegian format: +47 followed by 8 digits
- Amounts in NOK, realistic range (100-50000)
- Organization numbers: 9 digits
- The prompt should sound natural, like a real accounting instruction
- Return ONLY valid JSON, no markdown code fences"""

    user_prompt = f"""\
Task type: {task_def.name}
Description: {task_def.description}

{task_def.gen_instruction}

Return a JSON object with exactly these keys:
- "prompt": the task prompt text in {lang_name}
- "expected": an object with the expected field values

Return ONLY the JSON object, nothing else."""

    client = _get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,  # high creativity for variety
            response_mime_type="application/json",
        ),
    )

    text = response.text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3].strip()

    result = json.loads(text)

    return {
        "prompt": result["prompt"],
        "language": language,
        "expected": result["expected"],
        "task_def": task_def,
    }
