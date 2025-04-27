"""Streamlit UI for HR Validator"""

import io
import streamlit as st
import pdfplumber
from hr_validator import evaluate_candidate, DEFAULT_MODEL

st.set_page_config(page_title="HR Validator", page_icon="🕵️‍♀️")
st.title("🕵️‍♀️ HR Validator – Posouzení shody kandidáta 📄🔍")

model_name = "gpt-4o-mini"

st.sidebar.header("🔧 Nastavení")
openai_key = st.sidebar.text_input("OpenAI API key:", type="password", value="")

QUESTION_THRESHOLD = 60


def extract_text_as_markdown(uploaded_file) -> str:
    if not uploaded_file:
        return ""

    markdown_lines: list[str] = []
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if line.isupper():
                    markdown_lines.append(f"## {line.title()}")
                elif line.startswith("- ") or line.startswith("* "):
                    markdown_lines.append(line)
                else:
                    markdown_lines.append(line)
            markdown_lines.append("")
    return "\n".join(markdown_lines)


with st.form("hr_form"):
    jd_text = st.text_area(
        "Popis pozice", placeholder="Vložte popis pracovní pozice…", height=200
    )
    cv_file = st.file_uploader("Životopis kandidáta (PDF)", type=["pdf"])
    submit = st.form_submit_button("Vyhodnotit", type="primary")

if submit:
    if not jd_text or not cv_file:
        st.warning("Prosím vložte popis pozice a CV v PDF.")
        st.stop()

    cv_text = extract_text_as_markdown(cv_file)
    with st.spinner("Analyzuji životopis…"):
        result = evaluate_candidate(
            jd_text, cv_text, model=model_name, openai_key=openai_key
        )
    st.success("Hotovo!")

    score = result.get("score", 0)
    st.metric("Skóre vhodnosti", f"{score} / 100")

    if result.get("tags"):
        st.subheader("🔖 Tags pro pozici")
        st.markdown(", ".join(result["tags"]))

    st.subheader("Vysvětlení hodnocení")
    st.markdown(result.get("explanation", "_Žádné vysvětlení._"))

    st.subheader("Proč by kandidátovi mohla pozice vyhovovat")
    st.markdown(result.get("motivation", "_Bez uvedeného důvodu._"))

    if score >= QUESTION_THRESHOLD and result.get("questions"):
        st.subheader("Doplňující otázky")
        for q in result["questions"]:
            st.markdown(f"- {q}")
