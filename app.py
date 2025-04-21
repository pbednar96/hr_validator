"""Streamlit UI for HR Validator"""
import io
import streamlit as st
import pdfplumber
from hr_validator import evaluate_candidate, DEFAULT_MODEL

st.set_page_config(page_title="HR Validator", page_icon="🕵️‍♀️")
st.title("🕵️‍♀️ HR Validator – Posouzení shody kandidáta 📄🔍")

model_name = "gpt-4o-mini"

QUESTION_THRESHOLD = 60


def get_pdf_text(uploaded_file) -> str:
    """Read PDF bytes from Streamlit uploader and return extracted text."""
    if not uploaded_file:
        return ""
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)


with st.form("hr_form"):
    jd_text = st.text_area("Popis pozice", placeholder="Vložte popis pracovní pozice…", height=200)
    cv_file = st.file_uploader("Životopis kandidáta (PDF)", type=["pdf"])
    submit = st.form_submit_button("Vyhodnotit", type="primary")

if submit:
    if not jd_text or not cv_file:
        st.warning("Prosím vložte popis pozice a CV v PDF.")
        st.stop()

    cv_text = get_pdf_text(cv_file)
    with st.spinner("Analyzuji životopis…"):
        result = evaluate_candidate(jd_text, cv_text, model=model_name)
    st.success("Hotovo!")

    score = result.get("score", 0)
    st.metric("Skóre vhodnosti", f"{score} / 100")

    st.subheader("Vysvětlení hodnocení")
    st.markdown(result.get("explanation", "_Žádné vysvětlení._"))

    st.subheader("Proč by kandidátovi mohla pozice vyhovovat")
    st.markdown(result.get("motivation", "_Bez uvedeného důvodu._"))

    if score >= QUESTION_THRESHOLD and result.get("questions"):
        st.subheader("Doplňující otázky")
        for q in result["questions"]:
            st.markdown(f"- {q}")