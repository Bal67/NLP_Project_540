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
        
        # Load the tokenizer
        with open('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
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
        return text_padded
    except Exception as e:
        st.error(f"Error preprocessing text: {e}")
        return None

# Map time of day to numerical values
def map_time_of_day(time_of_day):
    mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    return mapping.get(time_of_day, 0)

# Function to predict sentiment
def predict_sentiment(text, time_of_day):
    preprocessed_text = preprocess_text(text)
    time_of_day_num = map_time_of_day(time_of_day)
    time_of_day_array = np.array([[time_of_day_num]])

    if preprocessed_text is None:
        return None

    try:
        prediction = model.predict([preprocessed_text, time_of_day_array])
        return prediction[0][0]
    except Exception as e:
        st.error(f"Error predicting sentiment: {e}")
        return None

# Streamlit application
st.title("Sentiment Analysis Application")
st.write("This application predicts the sentiment of the input text using an NLP model.")

# Initialize session state
if "text_input" not in st.session_state:
    st.session_state["text_input"] = ""
if "time_of_day" not in st.session_state:
    st.session_state["time_of_day"] = "Morning"

# Text input and time of day selection
text_input = st.text_area("Enter your text here:", value=st.session_state["text_input"])
time_of_day = st.selectbox("Select the time of day:", ["Morning", "Afternoon", "Evening", "Night"], index=["Morning", "Afternoon", "Evening", "Night"].index(st.session_state["time_of_day"]))

if st.button("Predict Sentiment"):
    # Save input values to session state
    st.session_state["text_input"] = text_input
    st.session_state["time_of_day"] = time_of_day
    
    if text_input:
        prediction = predict_sentiment(text_input, time_of_day)
        if prediction is not None:
            sentiment = "Positive" if prediction > 0.5 else "Negative"
            st.write(f"### NLP Model Prediction: {sentiment}")
            st.write(f"Positive Sentiment: {prediction*100:.2f}%")
            st.write(f"Negative Sentiment: {(1-prediction)*100:.2f}%")
        else:
            st.error("Prediction failed.")
    else:
        st.error("Please enter some text to analyze.")
