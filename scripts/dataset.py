## This file deals with importing dataset and subsequent cleaning and organizing of df

#Importing necessary libraries

import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report


#Dataset comes from Kaggle: 'Sentiment140 dataset with 1.6 million tweets'
#https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download
#Download dataset csv and save to location on local computer

#Importing path to CSV datafile
#path = '/content/drive/MyDrive/TextSentiment/training.1600000.processed.noemoticon.csv' #Replace with your own path to CSV data file in Google Drive

#Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

def load_dataset(path):
    """Load the dataset from the given path."""
    df = pd.read_csv(path, header=None, encoding='latin-1')
    return df

# Define stop words and stemmer
stop_words = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer('english')

# Define text cleaning regex
text_cleaning_re = r"@\S+|https?:\S+|http?:\S|[^A-Za-z0-9\s]+"

def tweet_cleaner(text):
    """Basic text cleaning by removing hyperlinks, stopwords, and applying stemming."""
    text = re.sub(text_cleaning_re,'', str(text).lower()).strip()
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words and len(word) >= 3]
    return " ".join(tokens)

# Define function to get time of day
def get_time_of_day(time):
    """Classify time into time of day."""
    hour = time.hour
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

def preprocess_dataset(df):
    """Preprocess the dataset by renaming columns, converting dates, and reducing size."""
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'tweet']
    
    # Understanding the "Target" column better and replacing 4 --> 1 
    df['target'] = df['target'].replace(4, 1)
    
    # Converting date to datetime
    df['date'] = pd.to_datetime(df['date'])

    #Creating columns that separate the date from the time
    df['date_only'] = df['date'].dt.strftime('%Y-%m-%d')
    df['time_only'] = df['date'].dt.strftime('%H:%M:%S')
    
    # Classifying date by season and time by time of day
    df['time_of_day'] = df['date'].apply(get_time_of_day)
    
    # Cutting down dataset by random 3/4 due to large size
    df_pos = df[df['target'] == 1].iloc[:200000]
    df_neg = df[df['target'] == 0].iloc[:200000]
    df = pd.concat([df_pos, df_neg], axis=0)

    #Apply Tweet Cleaning function to the 'tweet' column
    df['cleaned_tweet'] = df['tweet'].apply(tweet_cleaner)

    print(df.head())

    return df

if __name__ == "__main__":
    path = '/content/drive/MyDrive/TextSentiment/training.1600000.processed.noemoticon.csv'  # Update with the correct path
    df = load_dataset(path)
    df = preprocess_dataset(df)
    df.to_csv('/content/drive/MyDrive/TextSentiment/NLP_Project_540/data/preprocessed_dataset.csv', index=False) #Update to your own Google Drive folder path
    print("Preprocessed data saved to /content/drive/MyDrive/TextSentiment/NLP_Project_540/data/preprocessed_dataset.csv")
    
   