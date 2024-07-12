import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import joblib

@st.cache_resource
def load_models_and_tokenizer():
    # Load the TensorFlow model
    model_1 = load_model('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_model.h5')
    
    # Load the Naive Bayes model
    model_2 = joblib.load('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/naive_model.joblib')
    
    # Load the tokenizer for the TensorFlow model
    with open('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Load the TF-IDF vectorizer for the Naive Bayes model
    vectorizer = joblib.load('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/vectorizer.joblib')
    
    return model_1, model_2, tokenizer, vectorizer

# Load models, tokenizer, and vectorizer only once
model_1, model_2, tokenizer, vectorizer = load_models_and_tokenizer()

# Function to preprocess input text for the TensorFlow model
def preprocess_text_for_tf(text):
    text_seq = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_seq, maxlen=100)
    return text_padded

# Function to preprocess input text for the Naive Bayes model
def preprocess_text_for_nb(text):
    return vectorizer.transform([text])

# Function to predict sentiment
def predict_sentiment(text):
    preprocessed_text_tf = preprocess_text_for_tf(text)
    preprocessed_text_nb = preprocess_text_for_nb(text)
    
    # Debugging: Display shapes of preprocessed text
    st.write(f"Shape of preprocessed_text_tf: {preprocessed_text_tf.shape}")
    st.write(f"Shape of preprocessed_text_nb: {preprocessed_text_nb.shape}")

    prediction_1 = model_1.predict(preprocessed_text_tf)
    prediction_2 = model_2.predict(preprocessed_text_nb)

    # Debugging: Display predictions
    st.write(f"Prediction from model_1: {prediction_1}")
    st.write(f"Prediction from model_2: {prediction_2}")

    prediction_1 = prediction_1[0][0]
    prediction_2 = prediction_2[0]

    return prediction_1, prediction_2

# Streamlit application
st.title("Sentiment Analysis Application")
st.write("This application predicts the sentiment of the input text using two different models.")

text_input = st.text_area("Enter your text here:")
time_of_day = st.selectbox("Select the time of day:", ["Morning", "Afternoon", "Evening", "Night"])

if st.button("Predict Sentiment"):
    if text_input:
        pred_1, pred_2 = predict_sentiment(text_input)
        st.write(f"### TensorFlow Model Prediction:")
        st.write(f"Positive Sentiment: {pred_1*100:.2f}%")
        st.write(f"Negative Sentiment: {(1-pred_1)*100:.2f}%")
        
        st.write(f"### Naive Bayes Model Prediction:")
        st.write(f"Positive Sentiment: {pred_2*100:.2f}%")
        st.write(f"Negative Sentiment: {(1-pred_2)*100:.2f}%")
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
