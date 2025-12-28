import streamlit as st
import joblib
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Email Sentiment Predictor",
    page_icon="📧",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "senti_logreg_model.pkl"))

model = load_model()

# ---------------- UI ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    font-size: 18px;
    color: #b3b3b3;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    font-size: 22px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>📧 Email Sentiment Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Type an email and instantly understand its sentiment</div>", unsafe_allow_html=True)

st.write("")

email_text = st.text_area(
    "✍️ Enter Email Content",
    height=180,
    placeholder="Example: Thank you for your quick response. I really appreciate your support."
)

# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Sentiment"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter some email content.")
    else:
        try:
            prediction = model.predict([email_text])[0].lower()

            if prediction == "positive":
                st.markdown(
                    "<div class='result-box' style='background-color:#0f5132;color:#d1e7dd;'>😊 POSITIVE SENTIMENT</div>",
                    unsafe_allow_html=True
                )
                st.info("💡 **Suggestion:** Keep the tone friendly and appreciative. This email builds positive communication.")

            elif prediction == "negative":
                st.markdown(
                    "<div class='result-box' style='background-color:#842029;color:#f8d7da;'>😞 NEGATIVE SENTIMENT</div>",
                    unsafe_allow_html=True
                )
                st.warning("💡 **Suggestion:** Consider softening the language, adding polite phrases, or clarifying intent.")

            else:
                st.markdown(
                    "<div class='result-box' style='background-color:#41464b;color:#e2e3e5;'>😐 NEUTRAL SENTIMENT</div>",
                    unsafe_allow_html=True
                )
                st.info("💡 **Suggestion:** You may add warmth or clarity depending on the context.")

        except Exception as e:
            st.error("❌ Prediction failed. Please check model compatibility.")
