import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time
import streamlit.components.v1 as components
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Background Animation
# ------------------------------
def load_animation():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    animation_path = os.path.join(current_dir, "animation.html")
    
    if os.path.exists(animation_path):
        with open(animation_path, 'r') as f:
            animation_html = f.read()
    else:
        # Fallback inline HTML if file doesn't exist
        animation_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { margin: 0; padding: 0; width: 100vw; height: 100vh; overflow: hidden; background: transparent; pointer-events: none; }
                .word { position: absolute; color: rgba(40, 116, 166, 0.15); font-family: 'Helvetica Neue', Arial, sans-serif; font-weight: bold; user-select: none; pointer-events: none; animation: enlargeAndFade linear forwards; }
                @keyframes enlargeAndFade { 0% { transform: scale(0.5); opacity: 0; } 20% { opacity: 0.8; } 80% { opacity: 0.8; } 100% { transform: scale(2); opacity: 0; } }
            </style>
        </head>
        <body>
            <script>
                const words = ['NEWS', 'REPORT', 'BREAKING', 'UPDATE', 'HEADLINE', 'POLITICS', 'WORLD', 'BUSINESS', 'TECH', 'SCIENCE', 'HEALTH', 'SPORTS', 'ARTICLE', 'MEDIA', 'JOURNALISM', 'TRUTH', 'FACT', 'HOAX', 'SOURCE', 'VERIFY', 'STORY', 'PRESS', 'COVERAGE', 'EXCLUSIVE', 'INVESTIGATION'];
                function createWord() {
                    const wordEl = document.createElement('div');
                    wordEl.className = 'word';
                    wordEl.innerText = words[Math.floor(Math.random() * words.length)];
                    wordEl.style.top = (Math.random() * 80 + 10) + '%';
                    wordEl.style.left = (Math.random() * 80 + 10) + '%';
                    const size = Math.random() * 28 + 12;
                    wordEl.style.fontSize = size + 'px';
                    const duration = Math.random() * 3 + 3;
                    wordEl.style.animationDuration = duration + 's';
                    document.body.appendChild(wordEl);
                    setTimeout(() => { if (wordEl.parentNode) { wordEl.parentNode.removeChild(wordEl); } }, duration * 1000);
                }
                for (let i = 0; i < 3; i++) { setTimeout(createWord, i * 500); }
                setInterval(createWord, 800);
            </script>
        </body>
        </html>
        """
    
    return animation_html

# Embed the animation
components.html(load_animation(), height=0, scrolling=False)

# Add CSS to position the animation behind content
st.markdown("""
<style>
    iframe[title="streamlit_components_v1.html"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        z-index: -1 !important;
        pointer-events: none !important;
        border: none !important;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(1px);
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------
# Custom CSS for styling
# ------------------------------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# You would create a style.css file for this, but for simplicity, we'll embed it.
st.markdown("""
<style>
    /* Make main app background slightly transparent to see animation */
    .main {
        background-color: rgba(245, 245, 245, 0.8);
    }
    h1 {
        color: #2E4053;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #2874A6;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        font-size: 16px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1F618D;
    }
    .stTextArea textarea {
        border: 2px solid #AED6F1;
        border-radius: 8px;
        background-color: #FBFCFC;
        color: #000000 !important;
        font-size: 16px;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .stTextArea textarea::placeholder {
        color: #666666;
    }
    .result-box {
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .fake-news {
        background-color: #FADBD8;
        color: #C0392B;
        border: 2px solid #C0392B;
    }
    .real-news {
        background-color: #D5F5E3;
        color: #229954;
        border: 2px solid #229954;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------
# Load models + vectorizer
# ------------------------------
@st.cache_resource
def load_models_vectorizer():
    with open(r"C:\Users\himan\Downloads\Learning\Projects\Fake News Detection\model\lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    
    with open(r"C:\Users\himan\Downloads\Learning\Projects\Fake News Detection\model\naive_bayes_model.pkl", "rb") as f:
        nb_model = pickle.load(f)
    
    with open(r"C:\Users\himan\Downloads\Learning\Projects\Fake News Detection\model\tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    return lr_model, nb_model, vectorizer

lr_model, nb_model, vectorizer = load_models_vectorizer()

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("About the App")
st.sidebar.info(
    "This application uses Machine Learning models (Logistic Regression & Naive Bayes) "
    "to predict whether a news article is likely to be fake or real. "
    "The models were trained on a dataset of real and fake news articles."
)
st.sidebar.title("How to Use")
st.sidebar.markdown(
    "1. **Choose Model**: Select either Logistic Regression or Naive Bayes.\n"
    "2. **Enter Text**: Paste the news article or headline into the text box.\n"
    "3. **Predict**: Click the 'Analyze News' button.\n"
    "4. **View Result**: The app will classify the news as 'Real' or 'Fake' with confidence score."
)

# ------------------------------
# Main App
# ------------------------------
st.title("📰 Fake News Detection App")
st.markdown("This app allows you to test news articles/headlines with **Machine Learning models** to check if they are Fake or Real.")

# User selects model
model_choice = st.selectbox("Choose Model:", ["Logistic Regression", "Naive Bayes"])

# User Input
user_input = st.text_area("Enter News Text Here:", height=200)

if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing...'):
            time.sleep(1) # Simulate processing time
            # Transform input
            input_tfidf = vectorizer.transform([user_input])

            # Choose model
            if model_choice == "Logistic Regression":
                prediction = lr_model.predict(input_tfidf)[0]
                probability = lr_model.predict_proba(input_tfidf)[0][prediction]
            else:
                prediction = nb_model.predict(input_tfidf)[0]
                probability = nb_model.predict_proba(input_tfidf)[0][prediction]

            # Display result using custom styled boxes with confidence
            if prediction == 1:
                st.markdown(f'<div class="result-box fake-news">🚨 This looks like Fake News.<br><small>Confidence: {probability*100:.2f}%</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box real-news">✅ This looks like Real News.<br><small>Confidence: {probability*100:.2f}%</small></div>', unsafe_allow_html=True)