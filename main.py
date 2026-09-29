import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from preprocess_functions import *

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

    model = DecisionTreeClassifier(random_state=42)

    model.fit(X_train_final, y_train)

    y_pred = model.predict(X_val_final)

    accuracy = accuracy_score(y_val, y_pred)

    print("Accuracy:", accuracy)

    print(classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)

    print(cm)

    importances = model.feature_importances_

    for feature, importance in zip(features, importances):
        print(feature, importance)

if __name__ == "__main__":
    main()