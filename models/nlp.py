import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SpatialDropout1D, LSTM
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

sys.path.insert(0, '/content/drive/MyDrive/TextSentiment/NLP_Project_540/scripts')
from dataset import preprocess_dataset

def encode_time_of_day(time_of_day):
    time_mapping = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
    return np.array([time_mapping[time_of_day]])

def train_nlp_model():
    # Load and preprocess dataset
    path = '/content/drive/MyDrive/TextSentiment/NLP_Project_540/data/preprocessed_dataset.csv'  # Update with the correct path
    df = pd.read_csv(path)

    # Ensure all entries in 'cleaned_tweet' are strings and handle missing values
    df['cleaned_tweet'] = df['cleaned_tweet'].astype(str).fillna('')
    
    tokenizer = Tokenizer(num_words=5000, lower=True)
    tokenizer.fit_on_texts(df['cleaned_tweet'])
    X_seq = tokenizer.texts_to_sequences(df['cleaned_tweet'])
    X_padded = pad_sequences(X_seq, maxlen=100)

    # Encode time of day
    X_time = np.array([encode_time_of_day(tod) for tod in df['time_of_day']])
    X_combined = np.concatenate((X_padded, X_time), axis=1)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_combined, df['target'], test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    model = Sequential()
    model.add(Embedding(5000, 128, input_length=101))  # Adjust input length
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), verbose=2)
    
    # Save the tokenizer to Google Drive
    tokenizer_save_path = '/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_tokenizer.pkl'
    with open(tokenizer_save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Evaluate the model on the test set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(f'NLP Model Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

    # Save the model to Google Drive
    model.save('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_model.h5')
        
    return model, tokenizer

if __name__ == "__main__":
    train_nlp_model()
