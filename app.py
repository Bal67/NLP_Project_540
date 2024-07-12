import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

@st.cache_resource
def load_model_and_tokenizer():
    try:
        # Load the TensorFlow model
        model = load_model('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_model.h5')
        st.write("Model loaded successfully.")
        
        # Load the tokenizer
        with open('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        st.write("Tokenizer loaded successfully.")
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

# Load model and tokenizer only once
model, tokenizer = load_model_and_tokenizer()

# Function to preprocess input text for the NLP model
def preprocess_text(text):
    try:
        text_seq = tokenizer.texts_to_sequences([text])
        text_padded = pad_sequences(text_seq, maxlen=100)
        st.write(f"Preprocessed text: {text_padded}")
        return text_padded
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
        st.write(f"Raw prediction: {prediction}")
        return prediction[0][0]
    except Exception as e:
        st.error(f"Error predicting sentiment: {e}")
        return None

# Streamlit application
st.title("Sentiment Analysis Application")
st.write("This application predicts the sentiment of the input text using an NLP model.")

text_input = st.text_area("Enter your text here:")
time_of_day = st.selectbox("Select the time of day:", ["Morning", "Afternoon", "Evening", "Night"])

if st.button("Predict Sentiment"):
    if text_input:
        prediction = predict_sentiment(text_input)
        if prediction is not None:
            sentiment = "Positive" if prediction > 0.5 else "Negative"
            st.write(f"### NLP Model Prediction: {sentiment}")
            st.write(f"Positive Sentiment: {prediction*100:.2f}%")
            st.write(f"Negative Sentiment: {(1-prediction)*100:.2f}%")
        else:
            st.error("Prediction failed.")
    else:
        st.error("Please enter some text to analyze.")
