import streamlit as st
import joblib

@st.cache_resource
def load_model_and_vectorizer():
    try:
        # Load the Naive Bayes model
        model = joblib.load('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/naive_model.joblib')
        
        # Load the TF-IDF vectorizer
        vectorizer = joblib.load('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/vectorizer.joblib')
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load model and vectorizer only once
model, vectorizer = load_model_and_vectorizer()

# Function to preprocess input text for the Naive Bayes model
def preprocess_text(text):
    try:
        return vectorizer.transform([text])
    except Exception as e:
        st.error(f"Error preprocessing text: {e}")
        return None

# Function to predict sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    if preprocessed_text is None:
        return None

    try:
        prediction = model.predict(preprocessed_text)
        return prediction[0]
    except Exception as e:
        st.error(f"Error predicting sentiment: {e}")
        return None

# Streamlit application
st.title("Sentiment Analysis Application")
st.write("This application predicts the sentiment of the input text using a Naive Bayes model.")

text_input = st.text_area("Enter your text here:")
time_of_day = st.selectbox("Select the time of day:", ["Morning", "Afternoon", "Evening", "Night"])

if st.button("Predict Sentiment"):
    if text_input:
        prediction = predict_sentiment(text_input)
        if prediction is not None:
            sentiment = "Positive" if prediction == 1 else "Negative"
            st.write(f"### Naive Bayes Model Prediction: {sentiment}")
        else:
            st.error("Prediction failed.")
    else:
        st.error("Please enter some text to analyze.")

# Make the app visually attractive
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(to right, #ffffff, #e6e6e6);
        color: #000000;
    }
    .sidebar .sidebar-content {
        background: #f0f0f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)
