
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, SpatialDropout1D
from tensorflow.python.keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

sys.path.insert(0, '/content/drive/MyDrive/TextSentiment/NLP_Project_540/scripts/dataset.py')

def train_nlp_model():
    # Load and preprocess dataset
    path = '/content/drive/MyDrive/TextSentiment/NLP_Project_540/data/preprocessed_dataset.csv'  # Update with the correct path
    df = pd.read_csv(path)

    X = df['cleaned_tweet']
    y = df['target']
    
    tokenizer = Tokenizer(num_words=5000, lower=True)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_seq, maxlen=100)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_padded, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    model = Sequential()
    model.add(Embedding(5000, 128, input_length=100))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_val, y_val), verbose=2)
    
    model.save('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_model.h5')

    # Evaluate the model on the test set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(f'NLP Model Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
        
    return model, tokenizer

if __name__ == "__main__":
    train_nlp_model()

