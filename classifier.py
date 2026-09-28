import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import seaborn as sns

from preprocess_functions import *

df = pd.read_csv("./data/train_test_network.csv", sep=",")

features = [
    "dst_port",
    "proto",
    "duration"
]

X = df[features]
y = df["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

categorical_features, numerical_features = identify_features(df, features)

print(categorical_features)
print(numerical_features)

encoder.fit(X_train[categorical_features])

X_train_encoded = encoder.transform(X_train[categorical_features])
X_val_encoded = encoder.transform(X_val[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

X_train_final = np.concatenate([
    X_train[numerical_features].values,
    X_train_encoded
], axis=1)

X_val_final = np.concatenate([
    X_val[numerical_features].values,
    X_val_encoded
], axis=1)

X_test_final = np.concatenate([
    X_test[numerical_features].values,
    X_test_encoded
], axis=1)

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