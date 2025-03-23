from model.randomforest import RandomForest
from embeddings import get_tfidf_embeddings
from modelling.data_model import Data
import pandas as pd
from model.logistic import Logistic

# RandomForest
def model_predict(data, df, name):
    results = []
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)


# 
def train_chained_model(df):
    """
    Train a chained multi-output classification model:
    Step 1: Predict y2
    Step 2: Predict y3 (only on samples with y3)
    Step 3: Predict y4 (only on samples with y4)
    """

    label_columns = ['y2', 'y3', 'y4']

    for label in label_columns:
        print(f"\n--- Training model for: {label} ---")

        # Filter out rows with missing label
        df_label = df[df[label].notna() & (df[label] != '')].copy()
        df_label['y'] = df_label[label]  # Set current label column as 'y'

        # Convert email text to TF-IDF vectors
        X, y, df_label = get_tfidf_embeddings(df_label, 'y')

        # Prepare training and testing data
        data = Data(X, pd.DataFrame({'y': y}))

        # If not enough valid samples, skip
        if data.X_train is None:
            print(f"Skipping {label} due to insufficient data.")
            continue

        # Initialize and train model
        # 1.RandomForest
        model = RandomForest(model_name=f"RandomForest_{label}", embeddings=X, y=y)
        # 2.LogisticRegression
        # model = Logistic(model_name=f"Logistic_{label}", embeddings=X, y=y)

        model.train(data)

        # Predict and print evaluation results
        model.predict(data.X_test)
        model.print_results(data)
