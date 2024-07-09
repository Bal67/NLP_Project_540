import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

#Loading functions from dataset.py 
from dataset import load_dataset, preprocess_dataset

# Define the path to the dataset
path = '/Users/britt/Documents/Kaggle/training.1600000.processed.noemoticon.csv'  # Replace with your own path

# Load and preprocess the dataset
df = load_dataset(path)
df = preprocess_dataset(df)

#Remove unnessecary columns 
df = df.drop(['ids', 'date', 'flag', 'user', 'date_only', 'time_only'], axis=1)

#Check to see if equal number of 1 and 0 values
#print(df['target'].value_counts())

#print unique values in season
#print(df['time_of_day'].unique())

# Analyze word frequencies for target == 1 and target == 0
def analyze_word_frequencies(df):
    """Analyze word frequencies for positive and negative tweets."""
    positive_tweets = df[df['target'] == 1]['cleaned_tweet']
    negative_tweets = df[df['target'] == 0]['cleaned_tweet']
    
    positive_words = " ".join(positive_tweets).split()
    negative_words = " ".join(negative_tweets).split()
    
    positive_freq = Counter(positive_words)
    negative_freq = Counter(negative_words)
    
    print("Most common words in positive tweets:")
    print(positive_freq.most_common(10))
    
    print("\nMost common words in negative tweets:")
    print(negative_freq.most_common(10))

#Function to check for relationships between columns
def analyze_relationships(df):
    """Analyze relationships between columns"""

    df['target'] = df['target'].map({0: 'Negative', 1: 'Positive'})

    # Plot the distribution of tweets by target and time_of_day
    plt.figure(figsize=(14, 6))
    sns.countplot(data=df, x='time_of_day', hue='target')
    plt.title('Distribution of Tweets by Time of Day and Sentiment')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Tweets')
    plt.legend(title='Sentiment', loc='upper right', labels=['Negative', 'Positive'])
    plt.show()


# Call the function to analyze word frequencies
analyze_word_frequencies(df)

# Call the function to analyze relationships
analyze_relationships(df)