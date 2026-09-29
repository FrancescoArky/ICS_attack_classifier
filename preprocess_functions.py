import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def identify_features(df, features):
    categorical_features = []
    numerical_features = []

    for feature in features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            numerical_features.append(feature)
        else:
            categorical_features.append(feature)

    return categorical_features, numerical_features

def data_split(X, y):
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

    return X_train, y_train, X_val, y_val, X_test, y_test

def data_encoding(X_train, X_val, X_test, categorical_features, numerical_features):

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

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

    return X_train_final, X_val_final, X_test_final