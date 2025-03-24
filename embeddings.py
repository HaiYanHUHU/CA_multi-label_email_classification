
import numpy as np
import pandas as pd
from Config import *
import random
from sklearn.feature_extraction.text import TfidfVectorizer

# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)

# Convert full email (summary + content) into TF-IDF features
def get_tfidf_embd(df: pd.DataFrame):
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    
    # Combine summary and content
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    X = tfidfconverter.fit_transform(data).toarray()

    return X

# Concatenate two embeddings into one vector
def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)

# Convert Interaction Content into TF-IDF for a specific label
def get_tfidf_embeddings(df: pd.DataFrame, label_col: str):
    """
    Convert email content into TF-IDF vectors for a specific label column.
    :param df: Preprocessed DataFrame
    :param label_col: Label column to predict ('y2', 'y3', or 'y4')
    :return: X (TF-IDF matrix), y (label array), cleaned df
    """

    # Remove rows where the target label is missing or empty
    df = df[df[label_col].notna() & (df[label_col] != '')]

    # Extract text and label columns
    texts = (df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]).values.astype('U')
    labels = df[label_col].values.astype('U')

    # Transform text into TF-IDF vectors
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(texts)

    # Return vectors, labels and cleaned dataframe
    return X, labels, df
