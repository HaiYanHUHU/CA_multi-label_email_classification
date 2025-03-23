from model.randomforest import RandomForest
from embeddings import get_tfidf_embeddings
from modelling.data_model import Data
import pandas as pd
import numpy as np
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
    Train a chained multi-output classification model with step-wise evaluation:
    Step 1: Train and predict y2
    Step 2: Use correctly predicted y2 samples to train and predict y3
    Step 3: Use correctly predicted y2 + y3 samples to train and predict y4
    """

    labels = ['y2', 'y3', 'y4']
    predictions, truths, indices = {}, {}, {}     

    for label in labels:
        print(f"\n--- Training model for: {label} ---")
        df_label = df[df[label].notna() & (df[label] != '')].copy()
        df_label['y'] = df_label[label]

        X, y, df_label = get_tfidf_embeddings(df_label, 'y')

        #1.RandomForest
        model = RandomForest(model_name=f"RandomForest_{label}", embeddings=X, y=y)
        # 2.LogisticRegression
        #model = Logistic(model_name=f"Logistic_{label}", embeddings=X, y=y)

        X_transformed = model.data_transform(X)
        data = Data(X_transformed, pd.DataFrame({'y': y}))

        if data.X_train is None:
            print(f"Skipping {label} due to insufficient data.")
            continue
        
        model.train(data)
        model.predict(data.X_test)

        predictions[label] = model.predictions
        truths[label] = data.y_test
        indices[label] = np.arange(len(data.y_test))

        model.print_results(data)

    # evaluation
    print("\n Chained Evaluation ")
    final_scores = []

    # 
    for i, idx in enumerate(indices['y2']):
        score = 0
        correct_y2 = predictions['y2'][i] == truths['y2'][i]
        if not correct_y2:
            final_scores.append(0)
            continue

        score += 1
        idx_y3 = list(indices['y3']).index(idx) if idx in indices['y3'] else None
        if idx_y3 is None:
            final_scores.append(score / 3)
            continue

        correct_y3 = predictions['y3'][idx_y3] == truths['y3'][idx_y3]
        if not correct_y3:
            final_scores.append(score / 3)
            continue

        score += 1
        idx_y4 = list(indices['y4']).index(idx) if idx in indices['y4'] else None
        if idx_y4 is None:
            final_scores.append(score / 3)
            continue

        correct_y4 = predictions['y4'][idx_y4] == truths['y4'][idx_y4]
        if correct_y4:
            score += 1

        final_scores.append(score / 3)

    print(f"\nChained Multi-Output Final Accuracy: {np.mean(final_scores):.2%}")