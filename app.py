import streamlit as st
import pickle
import re
import nltk
import string

# ✅ ONLY FIX ADDED (DO NOT REMOVE)
nltk.download('stopwords')
nltk.download('punkt')

# ---------------- LOAD MODEL ----------------
with open("senti_logreg_model.pkl", "rb") as file:
    data = pickle.load(file)

model = data['model']
vectorizer = data['vectorizer']

# ---------------- ORIGINAL FUNCTIONS ----------------
def clean_text(text):
    text = text.lower()
    return text.strip()

def remove_punctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree

stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    output = " ".join(i for i in text.split() if i not in stopwords)
    return output

def remove_digits(text):
    clean_text_val = re.sub(r"\b[0-9]+\b\s*", "", text)
    return clean_text_val

def remove_emojis(data):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return re.sub(emoji_pattern, '', data)

# ---------------- PREPROCESS PIPELINE ----------------
def preprocess_email(text):
    text = clean_text(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = remove_digits(text)
    text = remove_emojis(text)
    return text

# ---------------- SENTIMENT LABEL ----------------
def get_sentiment_label(value):
    if value == -1:
        return "Negative 😟"
    elif value == 0:
        return "Neutral 😐"
    else:
        return "Positive 😊"

# ---------------- CLARITY CHECK ----------------
def clarity_check(text):
    if len(text.split()) < 6:
        return "Low ❌", "Email is too short and unclear."
    elif text.count('.') == 0:
        return "Medium ⚠️", "Try adding punctuation for better clarity."
    else:
        return "Good ✅", "Email is clear."

# ---------------- SUGGESTIONS ----------------
def suggestion(sentiment):
    if sentiment == "Negative 😟":
        return "Try using polite words like 'please', 'kindly', or 'could you'."
    elif sentiment == "Neutral 😐":
        return "You can improve the tone by adding a friendly closing line."
    else:
        return "Your email sounds polite and professional."

# ---------------- SESSION TRACKING ----------------
if "negative_count" not in st.session_state:
    st.session_state.negative_count = 0

if "unclear_count" not in st.session_state:
    st.session_state.unclear_count = 0

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="SentiMail", page_icon="📧")

st.title("📧 SentiMail – Where Emotions Meet Emails")
st.write("Analyze email sentiment, clarity, and improve communication.")

email_text = st.text_area("✉️ Enter Email Content", height=180)

if st.button("Analyze Email"):
    if email_text.strip() == "":
        st.warning("Please enter an email message.")
    else:
        processed_text = preprocess_email(email_text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        sentiment = get_sentiment_label(prediction)
        clarity, clarity_msg = clarity_check(email_text)
        advice = suggestion(sentiment)

        if prediction == -1:
            st.session_state.negative_count += 1
        if clarity.startswith("Low"):
            st.session_state.unclear_count += 1

        st.subheader("🔍 Analysis Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Clarity Level:** {clarity}")
        st.info(clarity_msg)

        st.subheader("💡 Improvement Suggestion")
        st.success(advice)

        st.subheader("📊 User Communication Stats")
        st.write(f"😟 Negative Emails: {st.session_state.negative_count}")
        st.write(f"❓ Unclear Emails: {st.session_state.unclear_count}")

