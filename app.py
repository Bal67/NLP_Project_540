import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model_and_vectorizer():
    # Load the Naive Bayes model
    model = joblib.load('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/naive_model.joblib')
    
    # Load the TF-IDF vectorizer
    vectorizer = joblib.load('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/vectorizer.joblib')
    
    return model, vectorizer

# Load model and vectorizer only once
model, vectorizer = load_model_and_vectorizer()

# Function to preprocess input text for the Naive Bayes model
def preprocess_text(text):
    return vectorizer.transform([text])

# Function to predict sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    return prediction[0]

# Streamlit application
st.title("Sentiment Analysis Application")
st.write("This application predicts the sentiment of the input text using a Naive Bayes model.")

text_input = st.text_area("Enter your text here:")
time_of_day = st.selectbox("Select the time of day:", ["Morning", "Afternoon", "Evening", "Night"])

if st.button("Predict Sentiment"):
    if text_input:
        prediction = predict_sentiment(text_input)
        sentiment = "Positive" if prediction == 1 else "Negative"
        st.write(f"### Naive Bayes Model Prediction: {sentiment}")
    else:
        st.error("Please enter some text to analyze.")
