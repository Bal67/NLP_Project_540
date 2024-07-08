## This file deals with importing dataset and subsequent cleaning and organizing of df

#Importing necessary libraries

import re
import numpy as np
import pandas as pd
import nltk

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
#path = '/Users/britt/Documents/Kaggle/training.1600000.processed.noemoticon.csv' #Replace with your own path to CSV data file downloaded from Kaggle

def load_dataset(path):
    """Load the dataset from the given path."""
    df = pd.read_csv(path, header=None, encoding='latin-1')
    return df


def preprocess_dataset(df):
    """Preprocess the dataset by renaming columns, converting dates, and reducing size."""
    df.columns = ['target', 'ids', 'date', 'flag', 'user', 'tweet']
    
    # Understanding the "Target" column better and replacing 4 --> 1 
    df['target'] = df['target'].replace(4, 1)
    
    # Converting date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Cutting down dataset by 3/4 due to large size
    df_pos = df[df['target'] == 1].iloc[:200000]
    df_neg = df[df['target'] == 0].iloc[:200000]
    df = pd.concat([df_pos, df_neg], axis=0)
    
    return df


def sample_data(df, n=6):
    """Sample the dataframe."""
    return df.sample(n)


