import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SpatialDropout1D, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

sys.path.insert(0, '/content/drive/MyDrive/TextSentiment/NLP_Project_540/scripts/dataset.py')

# Define a function to create the model
def create_model(embedding_dim=128, lstm_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2):
    model = Sequential()
    model.add(Embedding(5000, embedding_dim, input_length=100))
    model.add(SpatialDropout1D(dropout_rate))
    model.add(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def train_nlp_model():
    # Load and preprocess dataset
    path = '/content/drive/MyDrive/TextSentiment/NLP_Project_540/data/preprocessed_dataset.csv'  # Update with the correct path
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
    
    # Wrap the Keras model with KerasClassifier
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=64, verbose=2)
    
    # Define the grid of hyperparameters to search
    param_grid = {
        'embedding_dim': [128, 256],
        'lstm_units': [50, 100, 150],
        'dropout_rate': [0.2, 0.3],
        'recurrent_dropout_rate': [0.2, 0.3]
    }
    
    # Perform grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=2)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    # Display the best hyperparameters and accuracy
    print(f"Best Hyperparameters: {grid_result.best_params_}")
    print(f"Best Validation Accuracy: {grid_result.best_score_}")

    # Evaluate the best model on the test set
    best_model = grid_result.best_estimator_.model
    y_pred = (best_model.predict(X_test) > 0.5).astype("int32")
    print(f'NLP Model Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
    
    # Save the best model
    best_model.save('/content/drive/MyDrive/TextSentiment/NLP_Project_540/models/nlp_model.h5')

    return best_model, tokenizer

if __name__ == "__main__":
    train_nlp_model()

