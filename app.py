import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="AI Resume Screening", page_icon="📄")

st.title("📄 AI Resume Screening & Job Match System")
st.write("Upload your resume PDF and paste the job description to get a match score.")

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_match_score(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def analyze_skills(resume_text, skills):
    matched = []
    missing = []
    for skill in skills:
        if skill in resume_text:
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description here")
skills_input = st.text_input(
    "Enter required skills (comma separated)",
    "python, machine learning, sql"
)

if st.button("Analyze Resume"):
    if resume_file is None:
        st.error("❌ Please upload a resume PDF")
    elif jd_text.strip() == "":
        st.error("❌ Please paste the job description")
    else:
        resume_text = extract_text_from_pdf(resume_file)
        resume_text_clean = clean_text(resume_text)
        jd_text_clean = clean_text(jd_text)

        score = get_match_score(resume_text_clean, jd_text_clean)

        skills = [s.strip().lower() for s in skills_input.split(",")]
        matched_skills, missing_skills = analyze_skills(resume_text_clean, skills)

        st.success("✅ Analysis Complete")
        st.metric("Match Score", f"{score}%")
        st.progress(score / 100)

        st.subheader("✅ Matched Skills")
        if matched_skills:
            for s in matched_skills:
                st.write("•", s)
        else:
            st.write("No matched skills")

        st.subheader("❌ Missing Skills")
        if missing_skills:
            for s in missing_skills:
                st.write("•", s)
        else:
            st.write("None!")

