import numpy as np
import pandas as pd
import random

from preprocess import get_input_data, de_duplication, noise_remover
from Config import Config
from modelling.modelling import train_chained_model

# Set random seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

# Load and combine input CSVs
def load_data():
    df = get_input_data()
    return df

# Clean the data: remove duplicates, templates, signatures
def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)
    return df

if __name__ == '__main__':
    # Step 1: Load data
    df = load_data()

    # Step 2: Preprocess data
    df = preprocess_data(df)

    # Step 3: Ensure text columns are in string format
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')

    # Step 4: Train Chained Multi-Output Model (y2 → y3 → y4)
    train_chained_model(df)
