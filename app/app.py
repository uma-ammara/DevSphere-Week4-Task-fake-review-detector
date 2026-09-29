# app.py - Fake Product Review Detector (Streamlit)

import re
import joblib
import numpy as np
import streamlit as st
from tensorflow import keras

# ---------- Page setup ----------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Overall app background */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    /* Header block */
    .header-box {
        text-align: center;
        padding: 28px 16px 20px 16px;
    }
    .header-box h1 {
        color: #f8fafc;
        font-size: 2.1rem;
        margin-bottom: 4px;
    }
    .header-box p {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #1e293b;
        color: #f1f5f9;
        border: 1px solid #334155;
        border-radius: 12px;
        font-size: 1rem;
    }
    .stTextArea textarea:focus {
        border: 1px solid #6366f1;
        box-shadow: 0 0 0 1px #6366f1;
    }

    /* Analyze button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 10px 0;
        border-radius: 10px;
        border: none;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        transform: translateY(-1px);
    }

    /* Result cards */
    .result-card {
        padding: 22px;
        border-radius: 14px;
        margin-top: 18px;
        text-align: center;
    }
    .result-fake {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
    }
    .result-genuine {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid #22c55e;
    }
    .result-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .result-fake .result-title { color: #f87171; }
    .result-genuine .result-title { color: #4ade80; }

    .prob-row {
        display: flex;
        justify-content: space-between;
        color: #cbd5e1;
        font-size: 0.9rem;
        margin-top: 12px;
    }

    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Load the saved model and vectorizer (only once) ----------
@st.cache_resource
def load_files():
    model = keras.models.load_model("fake_review_model.keras")
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    return model, tfidf

model, tfidf = load_files()

# ---------- Same cleaning function used in training ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- Header ----------
st.markdown("""
<div class="header-box">
    <h1>🕵️ Fake Review Detector</h1>
    <p>AI-powered neural network that spots suspicious product reviews</p>
</div>
""", unsafe_allow_html=True)

# ---------- Input ----------
review = st.text_area(
    "Enter a review to analyze:",
    height=170,
    placeholder="Paste or type a product/hotel review here..."
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_clicked = st.button("🔍 Analyze Review")

# ---------- Prediction and result ----------
if analyze_clicked:
    cleaned = clean_text(review)

    if len(cleaned) == 0:
        st.warning("⚠️ Please enter a review first.")
    else:
        with st.spinner("Analyzing review..."):
            features = tfidf.transform([cleaned]).toarray().astype("float32")
            prob_fake = float(model.predict(features, verbose=0)[0][0])
            prob_genuine = 1 - prob_fake

        is_fake = prob_fake >= 0.5
        confidence = prob_fake if is_fake else prob_genuine

        card_class = "result-fake" if is_fake else "result-genuine"
        icon = "⚠️" if is_fake else "✅"
        label = "Potentially Fake" if is_fake else "Likely Genuine"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-title">{icon} {label}</div>
            <div style="color:#e2e8f0; font-size:1.1rem;">
                Confidence: <b>{confidence:.1%}</b>
            </div>
            <div class="prob-row">
                <span>Genuine: {prob_genuine:.1%}</span>
                <span>Fake: {prob_fake:.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(prob_fake)

st.markdown('<div class="footer-note">This is a student project. Results are predictions, not proof.</div>',
           unsafe_allow_html=True)