import streamlit as st
import pickle
import re
import nltk
import string
from nltk.corpus import stopwords

# Download required NLTK data (one-time)
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

# Load model and vectorizer using YOUR exact filename
@st.cache_resource
def load_model():
    with open('sentimail_logreg_model.pkl', 'rb') as file:
        data = pickle.load(file)
        model = data['model']
        vectorizer = data['vectorizer']
    return model, vectorizer

model, vectorizer = load_model()

# Preprocessing functions (exact from your notebook)
def clean_text(text):
    text = text.lower()
    return text.strip()

def remove_punctuation(text):
    punctuation_free = ''.join(i for i in text if i not in string.punctuation)
    return punctuation_free

def remove_stopwords(text):
    stopwords_list = stopwords.words('english')
    output = ' '.join(i for i in text.split() if i not in stopwords_list)
    return output

def remove_digits(text):
    clean_text = re.sub(r'\d+', ' ', text)
    return clean_text

def remove_emojis(data):
    emojipattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return re.sub(emojipattern, ' ', data)

def preprocess_email(text):
    text = clean_text(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = remove_digits(text)
    text = remove_emojis(text)
    return text

# Sentiment label mapping (exact from your notebook)
def get_sentiment_label(value):
    if value == -1:
        return "Negative 😟"
    elif value == 0:
        return "Neutral 😐"
    else:
        return "Positive 🙂"

# Clarity and suggestions
def clarity_check(text):
    words = len(text.split())
    if words < 6:
        return "Low", "Email is too short and unclear."
    elif text.count('.') == 0:
        return "Medium", "Try adding punctuation for better clarity."
    else:
        return "Good", "Email is clear."

def suggestion(sentiment):
    if sentiment == "Negative":
        return "Try using polite words like 'please', 'kindly', or 'could you'."
    elif sentiment == "Neutral":
        return "Add a friendly closing line like 'Best regards'."
    else:
        return "Your email sounds polite and professional."

# Session state for stats
if 'negative_count' not in st.session_state:
    st.session_state.negative_count = 0
if 'unclear_count' not in st.session_state:
    st.session_state.unclear_count = 0

# UI
st.set_page_config(page_title="SentiMail", page_icon="📧")
st.title("📧 SentiMail")
st.write("Analyze email sentiment, clarity, and improve communication.")

email_text = st.text_area("Enter Email Content", height=180)

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

        # Update stats
        if prediction == -1:
            st.session_state.negative_count += 1
        if clarity.startswith("Low"):
            st.session_state.unclear_count += 1

        st.subheader("📊 Analysis Result")
        st.metric("Sentiment", sentiment)
        st.metric("Clarity Level", clarity)
        st.info(clarity_msg)

        st.subheader("💡 Improvement Suggestion")
        st.success(advice)

        st.subheader("📈 User Communication Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Negative Emails", st.session_state.negative_count)
        with col2:
            st.metric("Unclear Emails", st.session_state.unclear_count)

# Reset stats
if st.button("Reset Stats"):
    st.session_state.negative_count = 0
    st.session_state.unclear_count = 0
    st.rerun()
