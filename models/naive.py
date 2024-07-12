import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SpatialDropout1D, LSTM
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle

sys.path.insert(0, '/content/drive/MyDrive/TextSentiment/NLP_Project_540/scripts')
from dataset import preprocess_dataset

def train_non_fine_tuned_lstm_model():
    # Load and preprocess dataset
    path = '/content/drive/MyDrive/TextSentiment/NLP_Project_540/data/preprocessed_dataset.csv'  # Update with the correct path
    df = pd.read_csv(path)

    # Ensure all entries in 'cleaned_tweet' are strings and handle missing values
    df['cleaned_tweet'] = df['cleaned_tweet'].astype(str).fillna('')

    # Use a smaller subset for quick testing
    df = df.sample(5000, random_state=42)

    X = df['cleaned_tweet']
    y = df['target']
    
    tokenizer = Tokenizer(num_words=5000, lower=True)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_seq, maxlen=50)  # Reduced maxlen for faster processing
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_padded, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    model = Sequential()
    model.add(Embedding(5000, 64, input_length=50))  # Reduced embedding size
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))  # Further reduced LSTM units
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=2, batch_size=32, validation_data=(X_val, y_val), verbose=2)  # Reduced epochs for faster training
    
    # Save the tokenizer to Google Drive
    tokenizer_save_path = '/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/non_fine_tuned_lstm_tokenizer.pkl'
    with open(tokenizer_save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Evaluate the model on the test set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(f'Non-Fine-Tuned LSTM Model Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

    # Save the model to Google Drive
    model.save('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/non_fine_tuned_lstm_model.keras')
        
    return model, tokenizer

if __name__ == "__main__":
    train_non_fine_tuned_lstm_model()
