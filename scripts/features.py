import re
import numpy as np
import pandas as pd
import nltk

# plotting
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

