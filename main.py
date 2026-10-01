import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from preprocess_functions import *
from models import *

def main():

    df = pd.read_csv("./data/train_test_network.csv", sep=",")

    features = [
        "dst_port",
        "proto",
        "duration"
    ]

    X = df[features]
    y = df["label"]

    X_train, y_train, X_val, y_val, X_test, y_test = data_split(X, y)

    categorical_features, numerical_features = identify_features(df, features)

    X_train_final, X_val_final, X_test_final = data_encoding(X_train, X_val, X_test, categorical_features, numerical_features)

    decision_tree_classifier(features, X_train_final, y_train, X_val_final, y_val)

if __name__ == "__main__":
    main()