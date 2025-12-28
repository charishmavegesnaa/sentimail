import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Email Sentiment Data",
    page_icon="📧",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("senti_logreg_model.pkl")

model = load_model()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("Email_Sentiment_Data.csv")

df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Menu")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Dataset", "✍️ Predict", "📈 Model Details"]
)

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.title("📧 Email Sentiment Data")
    st.markdown("""
    ### Interactive Email Sentiment Analysis App
    - Uses trained ML model
    - Clean & balanced dataset
    - Real-time sentiment prediction
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Emails", len(df))
    col2.metric("Unique Senders", df["From Name"].nunique())
    col3.metric("Sentiment Classes", df["Sentiment"].nunique())

    st.success("✅ Application loaded successfully")

# ---------------- DATASET ----------------
elif page == "📊 Dataset":
    st.title("📊 Email Sentiment Dataset")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(25), use_container_width=True)

    st.subheader("Sentiment Distribution")
    st.bar_chart(df["Sentiment"].value_counts())

    st.subheader("Filter by Sentiment")
    sentiment = st.selectbox(
        "Choose sentiment",
        ["All"] + list(df["Sentiment"].unique())
    )

    if sentiment != "All":
        st.dataframe(
            df[df["Sentiment"] == sentiment],
            use_container_width=True
        )

# ---------------- PREDICTION ----------------
elif page == "✍️ Predict":
    st.title("✍️ Predict Email Sentiment")

    email_text = st.text_area(
        "Enter email content",
        height=180,
        placeholder="Type or paste email text here..."
    )

    if st.button("🔍 Predict Sentiment"):
        if email_text.strip() == "":
            st.warning("Please enter email text")
        else:
            result = model.predict([email_text])[0]

            if result.lower() == "positive":
                st.success("😊 POSITIVE sentiment")
            elif result.lower() == "negative":
                st.error("😞 NEGATIVE sentiment")
            else:
                st.info("😐 NEUTRAL sentiment")

# ---------------- MODEL DETAILS ----------------
elif page == "📈 Model Details":
    st.title("📈 Model Details")

    st.markdown("""
    **Model File:** `senti_logreg_model.pkl`  
    **Dataset:** `Email_Sentiment_Data.csv`  

    ### Key Points
    - Logistic Regression model
    - Trained on balanced dataset
    - Suitable for academic & demo purposes
    - Fast and interpretable
    """)

    st.success("Model ready for predictions 🚀")
