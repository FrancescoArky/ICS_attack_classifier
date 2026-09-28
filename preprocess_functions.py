import pandas as pd

def identify_features(df, features):
    categorical_features = []
    numerical_features = []

    for feature in features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            numerical_features.append(feature)
        else:
            categorical_features.append(feature)

    return categorical_features, numerical_features