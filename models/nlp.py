import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SpatialDropout1D, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
import numpy as np

sys.path.insert(0, '/content/drive/MyDrive/TextSentiment/NLP_Project_540/scripts/dataset.py')
MODEL_FILE = "/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_model.h5"
EMBEDDING_FILE = 'glove.6B.100d.txt'

def load_glove_embeddings():
    embeddings_index = {}
    with open(EMBEDDING_FILE, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def create_embedding_matrix(tokenizer, embeddings_index, embedding_dim=100):
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def train_nlp_model():
    # Load and preprocess dataset
    path = '/content/drive/MyDrive/TextSentiment/NLP_Project_540/data/preprocessed_dataset.csv'
    df = pd.read_csv(path)

    # Ensure all entries in 'cleaned_tweet' are strings and handle missing values
    df['cleaned_tweet'] = df['cleaned_tweet'].astype(str).fillna('')

    X = df['cleaned_tweet']
    y = df['target']
    
    tokenizer = Tokenizer(num_words=5000, lower=True)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_seq, maxlen=100)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_padded, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    embeddings_index = load_glove_embeddings()
    embedding_matrix = create_embedding_matrix(tokenizer, embeddings_index)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                        output_dim=100,
                        weights=[embedding_matrix],
                        input_length=100,
                        trainable=False))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    # Implement EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_val, y_val), verbose=2, callbacks=[early_stopping])

    # Save the tokenizer to Google Drive
    tokenizer_save_path = '/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_tokenizer.pkl'
    with open(tokenizer_save_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Evaluate the model on the test set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print(f'NLP Model Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

    # Save the model to Google Drive
    model.save(MODEL_FILE)
        
    return model, tokenizer

if __name__ == "__main__":
    train_nlp_model()
