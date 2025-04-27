"""Core utilities for HR Validator"""

import json
from typing import List, Dict, Any

import pdfplumber
import openai


DEFAULT_MODEL = "gpt-4.1"  # Adjust if different suffix when generally available


SYSTEM_MESSAGE = (
    "Jsi zkušený personalista a tvým úkolem je předběžně posoudit, zda se kandidát na základě svého "
    "životopisu hodí na danou pracovní pozici napříč obory (např. technické profese, IT, účetnictví, "
    "stavebnictví, energetika, výroba, administrativa aj.). "
    "Soustřeď se zejména na:\n"
    "• klíčové odborné požadavky a dovednosti (technické, soft-skills, legislativní či jazykové),\n"
    "• typ práce a pracovní prostředí (kancelář, terén, výroba, projekční činnost, zákaznická podpora …),\n"
    "• požadovanou úroveň praxe, vzdělání, certifikací nebo oprávnění (např. vyhláška 50/1978, ACCA, CAD licence, řidičské oprávnění atd.).\n"
    "Pokud kandidát působil převážně v jiném oboru, zohledni to negativně, ledaže je patrná logická "
    "motivace ke změně či přenositelné dovednosti. "
    "Chybějí-li v CV zásadní údaje (např. konkrétní technologie, nástroje, projekty, objem zakázek, "
    "odpovědnost za rozpočet, certifikace), vygeneruj doplňující otázky, které by personalista měl položit. "
    "Odpověz výhradně česky. Vrať JSON objekt s těmito klíči:\n"
    "- score: celé číslo 0–100 vyjadřující vhodnost kandidáta vůči pozici.\n"
    "- explanation: stručné odůvodnění uděleného skóre.\n"
    "- motivation: případný důvod, proč by pozice mohla kandidáta oslovit – uveď jen pokud dává smysl.\n"
    "- questions: pole doplňujících otázek; pokud nejsou potřeba, vrať prázdné pole.\n"
    "- tags: pole klíčových (dovedností/technologií/obdobne nazvy pozic) pro danou pracovní pozici (MAX 10 tagů). Nepřidávej dovednosti kandidáta – tagy musí vycházet pouze z požadavků pozice."
    "Příklady:\n"
    "  • Java Developer → [\"JAVA\", \"SPRING\", \"GIT\", \"SOFTWARE ENGINEER\", \"BACK-END DEVELOPER\"]\n"
    "  • Účetní → [\"IFRS\", \"SAP\", \"MS EXCEL\", \"ACCOUNTANT\", \"FINANČNÍ ÚČETNÍ\"]\n"
    "  • Architekt → [\"AUTOCAD\", \"REVIT\", \"BIM\", \"PROJEKTANT\", \"STAVEBNÍ ARCHITEKT\"]\n"
    "  • Elektrikář (slaboproud) → [\"VYHLÁŠKA_50\", \"SCHÉMATA\", \"MULTIMETR\", \"ELEKTROTECHNIK\", \"TECHNIK SLABOPROUD\"]\n"
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
        temperature=0.65,
        top_p=0.9
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
