"""Core utilities for HR Validator"""
import os
import json
from typing import List, Dict, Any

import pdfplumber
import openai

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"  # Adjust if different suffix when generally available

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_MESSAGE = (
    "Jsi zkušený personalista a tvým úkolem je předběžně posoudit, zda se kandidát na základě svého životopisu hodí na danou pracovní pozici. "
    "Zaměř se zejména na technologické požadavky, typ práce a pracovní zaměření pozice ve srovnání s dosavadními zkušenostmi kandidáta. "
    "Pokud kandidát pracoval v jiné oblasti (např. C++ vývojář vs. Python QA), zohledni to negativně, pokud není zjevný přechod nebo motivace. "
    "Pokud v životopisu chybí důležité informace (např. konkrétní technologie, zkušenosti s AI, testováním, apod.), vygeneruj otázky, které by měl personalista položit. "
    "Odpověz výhradně česky. Vrať JSON objekt s následujícími klíči:\n"
    "- score: celé číslo od 0 do 100, které vyjadřuje míru vhodnosti kandidáta na základě popisu pozice a CV.\n"
    "- explanation: stručné a výstižné odůvodnění, proč byl zvolen daný počet bodů.\n"
    "- motivation: případný důvod, proč by kandidáta mohla pozice zaujmout – pouze pokud to dává smysl.\n"
    "- questions: pole doplňujících otázek, které by personalista měl položit, pokud v CV chybí zásadní informace.\n"
    "Pokud nejsou potřeba žádné další otázky, vrať prázdné pole questions."
)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file path."""
    with pdfplumber.open(pdf_path) as pdf:
        pages_text = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages_text)


def build_messages(job_description: str, cv_text: str) -> List[Dict[str, str]]:
    """Prepare chat messages for the OpenAI ChatCompletion call."""
    user_content = (
        "<JOB_DESCRIPTION>\n" + job_description.strip() + "\n</JOB_DESCRIPTION>\n" +
        "<RESUME>\n" + cv_text.strip() + "\n</RESUME>"
    )
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content}
    ]


def _call_openai(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL, openai_key: str | None = None) -> Dict[str, Any]:
    openai.api_key = openai_key or os.getenv("OPENAI_API_KEY")
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def evaluate_candidate(job_description: str, cv_text: str, *, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    messages = build_messages(job_description, cv_text)
    return _call_openai(messages, model=model, openai_key=OPENAI_API_KEY)