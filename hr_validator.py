"""Core utilities for HR Validator"""

import json
from typing import List, Dict, Any

import pdfplumber
import openai


DEFAULT_MODEL = "gpt-4.1-mini"  # Adjust if different suffix when generally available


SYSTEM_MESSAGE = (
    "Jsi zkušený personalista. Na základě popisu pracovní pozice (JD) a životopisu kandidáta (CV) "
    "posuď míru shody. Nezohledňuj jméno, pohlaví ani osobní údaje – hodnotíš jen profesní vhodnost.\n\n"

    "Zaměř se na:\n"
    "1. Klíčové odborné požadavky a dovednosti (technické, legislativní, jazykové, soft-skills).\n"
    "2. Typ práce a prostředí (kancelář, terén, výroba, projekční činnost …).\n"
    "3. Požadovanou úroveň praxe, vzdělání a certifikací/oprávnění.\n"
    "4. Kontinuitu oboru; pokud kandidát přechází z jiné oblasti, zvaž přenositelné dovednosti a motivaci.\n\n"

    "Skórování (0–100 bodů):\n"
    "0-30 slabá shoda · 31-70 částečná shoda · 71-100 výborná shoda.\n\n"

    "Pokud v CV chybí zásadní fakta (konkrétní technologie, rozsah odpovědnosti, certifikace …), "
    "vygeneruj doplňující otázky.\n\n"

    "Výstup vrať **pouze** jako validní JSON bez dalších komentářů, v tomto pořadí klíčů:\n"
    "{\n"
    "  \"score\": <int 0-100>,\n"
    "  \"explanation\": \"<krátké odůvodnění max 50 slov>\",\n"
    "  \"motivation\": \"<důvod zájmu, nebo prázdný řetězec>\",\n"
    "  \"questions\": [<doplňující dotazy>],\n"
    "  \"skill_tags\": [<max 10 TECHNOLOGIÍ a CERTIFIKACÍ VELKÝMI PÍSMENY>],\n"
    "  \"role_tags\": [<max 10 SYNONYMNÍCH NÁZVŮ POZIC VELKÝMI PÍSMENY>]\n"
    "}\n\n"

    "Příklady tagů (pro ilustraci formátu, neužívej je bezdůvodně):\n"
    "• Java Developer → skill_tags [\"JAVA\", \"SPRING\", \"GIT\"], role_tags [\"SOFTWARE ENGINEER\", \"BACK-END DEVELOPER\"]\n"
    "• Účetní → skill_tags [\"IFRS\", \"SAP\", \"MS_EXCEL\"], role_tags [\"ACCOUNTANT\", \"FINANCIAL_ACCOUNTANT\"]\n"
    "• Architekt → skill_tags [\"AUTOCAD\", \"REVIT\", \"BIM\"], role_tags [\"PROJEKTANT\", \"STAVEBNI_ARCHITEKT\"]\n"
    "• Elektrikář (slaboproud) → skill_tags [\"VYHLASKA_50\", \"MULTIMETR\"], role_tags [\"ELEKTROTECHNIK\", \"TECHNIK_SLABOPROUD\"]\n\n"

    "Odpověz výhradně česky."
)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file path."""
    with pdfplumber.open(pdf_path) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages_text)


def build_messages(job_description: str, cv_text: str) -> List[Dict[str, str]]:
    """Prepare chat messages for the OpenAI ChatCompletion call."""
    user_content = (
        "<JOB_DESCRIPTION>\n"
        + job_description.strip()
        + "\n</JOB_DESCRIPTION>\n"
        + "<RESUME>\n"
        + cv_text.strip()
        + "\n</RESUME>"
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]


def _call_openai(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    openai_key: str | None = None,
) -> Dict[str, Any]:
    openai.api_key = openai_key
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def evaluate_candidate(
    job_description: str,
    cv_text: str,
    *,
    model: str = DEFAULT_MODEL,
    openai_key: str = None
) -> Dict[str, Any]:
    messages = build_messages(job_description, cv_text)
    return _call_openai(messages, model=model, openai_key=openai_key)
