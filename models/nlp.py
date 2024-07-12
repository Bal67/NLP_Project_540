import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dense, SpatialDropout1D, LSTM, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
import logging
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

sys.path.insert(0, '/content/drive/MyDrive/TextSentiment/NLP_Project_540/scripts/dataset.py')
MODEL_FILE = "/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_model.h5"

# Train the function
def train_nlp_model():
    # Load and preprocess dataset
    path = '/content/drive/MyDrive/TextSentiment/NLP_Project_540/data/preprocessed_dataset.csv'
    df = pd.read_csv(path)

    # For testing, limit the size of the dataset
    df = df.sample(frac=0.1, random_state=42)  # Use 10% of the data for testing

    # Ensure all entries in 'cleaned_tweet' are strings and handle missing values
    df['cleaned_tweet'] = df['cleaned_tweet'].astype(str).fillna('')

    # Convert time_of_day to numerical categories
    time_of_day_mapping = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
    df['time_of_day'] = df['time_of_day'].map(time_of_day_mapping)

    X_text = df['cleaned_tweet']
    X_time_of_day = df['time_of_day']
    y = df['target']
    
    # Adding a tokenizer
    tokenizer = Tokenizer(num_words=5000, lower=True)
    tokenizer.fit_on_texts(X_text)
    X_seq = tokenizer.texts_to_sequences(X_text)
    X_padded = pad_sequences(X_seq, maxlen=100)
    
    # Splitting the data into training, validation, and test sets
    X_train_text, X_temp_text, y_train, y_temp, X_train_time, X_temp_time = train_test_split(X_padded, y, X_time_of_day, test_size=0.4, random_state=42)
    X_val_text, X_test_text, y_val, y_test, X_val_time, X_test_time = train_test_split(X_temp_text, y_temp, X_temp_time, test_size=0.5, random_state=42)
    
    # Building the model
    text_input = Input(shape=(100,), name='text_input')
    time_input = Input(shape=(1,), name='time_input')

    embedding = Embedding(5000, 64, input_length=100)(text_input)
    dropout = SpatialDropout1D(0.2)(embedding)
    lstm = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(dropout)

    concatenated = Concatenate()([lstm, time_input])
    output = Dense(1, activation='sigmoid')(concatenated)

    model = Model(inputs=[text_input, time_input], outputs=output)
    
    # Compiling the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    # Implement EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit([X_train_text, X_train_time], y_train, epochs=10, batch_size=64, validation_data=([X_val_text, X_val_time], y_val), verbose=2, callbacks=[early_stopping])

    # Save the tokenizer to Google Drive
    tokenizer_save_path = '/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_tokenizer.pkl'
    with open(tokenizer_save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Evaluate the model on the test set
    y_pred = (model.predict([X_test_text, X_test_time]) > 0.5).astype("int32")
    print(f'NLP Model Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

    # Save the model to Google Drive
    model.save(MODEL_FILE)
        
    return model, tokenizer

if __name__ == "__main__":
    train_nlp_model()
