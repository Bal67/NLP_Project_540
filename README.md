# NLP_Project_540

Text Sentiment Classification Based on Time of Day

Google Colab: https://colab.research.google.com/drive/1ojBnhAwkWIASMqb0XgJRBeKZC0P14JJy?usp=sharing

Kaggle Dataset: https://www.kaggle.com/datasets/kazanova/sentiment140/code

This project classifies whether a textual response is positive or negative based on what time of day it is (Morning, Afternoon, Evening, and Night). The models used are a fine-tuned LSTM-based Neural Network and a non-fine-tuned LSTM model.


## Table of Contents

- [Setup](#setup)
- [Main](#main)
- [scripts](#scripts)
- [models](#models)
- [data](#data)

## Project Structure

setup.py: Script for setting up environment

app.py: The main Streamlit app

scripts/: Contains the scripts for generating graphs and processing data

dataset.py: Dataset loading and preprocessing

features.py: Processed features from the dataset

models/: Contains the trained models

naive.py: Trained, non-fine-tuned NLP model

nlp.py: The trained, fine-tuned NLP model

data/: Contains the dataset

requirements.txt: List of dependencies

README.md


## Usage

Use the Kaggle link to download the zipfile for the dataset (https://www.kaggle.com/datasets/kazanova/sentiment140/code) . Proceed to the Google Colab page that is linked at the top of this README.md. Once at the page, mount to your own Google Drive in order to and proceed to follow the instructions for each cell of the Google Colab. 

Replace all of the #Constants in the code (or anywhere where you see a pathway) with the pathway to your local Google Drive folder/Google Drive pathway

For the StreamLit application: Google Colab has a hard time opening StreamLit applications. In order to do so, you must run the final cell. At the bottom of that cell will be a link that will lead you to a tunnel website. The bottom cell will also provide you with an IP Address that will look as such (XX.XXX.XXX.XX). Insert that address into the tunnel when prompted for a passcode to access the StreamLit application.


# Model Evaluation


## Evaluation Process and Metric Selection

The evaluation process involves splitting the data into training, validation and testing sets (60-20-20), training the models, and then evaluating their performance on the test set. The primary metric used for evaluation is accuracy, which measures the proportion of correctly classified instances.

For the NLP model, additional metrics like loss and validation accuracy are also considered. In the context of this sentiment analysis project, accuracy represents the proportion of tweets for which the model correctly predicted the sentiment (positive or negative) out of the total number of tweets in the test set. A higher accuracy indicates that the model is more effective at distinguishing between positive and negative sentiments, thereby demonstrating its ability to understand and interpret the textual data accurately. 


## Data Processing Pipeline

Data Loading: Data is loaded into the script in CSV format

Feature Extraction: Data analyzed for relationships. Additional columns made for ”Time of Day”, “Clean Tweet”, and ”Positive/Negative Sentiment”. 

Data Preparation: Data is split into features and column labels are added. Null values taken out. Data split into training (60%), validation (20%), and testing sets (20%).

Model Training: Both the naive and NLP models are trained on the training data. Continued evaluation of validation set

Model Evaluation: Models are evaluated on the test data and accuracy recorded


## Models Evaluated

Naive Model: Non-fine-tuned LTSM Neural Network


NLP Model: Fine-tuned LTSM Neural Network

Architecture:

  -Embedding Layer

  -LSTM Layer

  -Dense Layer


## Results and Conclusions
Naive Model Accuracy: Achieved an accuracy of approximately 71.50% on the test set.

NLP Model Accuracy: Achieved a validation accuracy of approximately 74% on the test set.

The project demonstrates that both naive LSTM and fine-tuned LSTM models can classify the sentiment of texts, with the fine-tuned NLP model showing potential for further improvements with more data and tuning.


# Acknowledgments
Data sourced from Sentiment140.
This project was developed as part of a machine learning course/project.
