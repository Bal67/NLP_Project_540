# naive.py

from scripts.dataset import load_dataset, preprocess_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

def train_naive_model():
    # Load and preprocess dataset
    path = '/Users/britt/Documents/Kaggle/training.1600000.processed.noemoticon.csv'  # Update with the correct path
    df = load_dataset(path)
    df = preprocess_dataset(df)

    X = df['cleaned_tweet']
    y = df['target']
    
    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_vect, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'naive_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    print(f'Naive Bayes Model Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

    return model, vectorizer

if __name__ == "__main__":
    train_naive_model()
