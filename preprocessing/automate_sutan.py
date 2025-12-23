import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)

class DateTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column="TransactionDate"):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column])
        X["hour"] = X[self.date_column].dt.hour
        X["day_of_week"] = X[self.date_column].dt.dayofweek
        X["is_weekend"] = X["day_of_week"].isin([5, 6]).astype(int)
        X.drop(columns=[self.date_column], inplace=True)
        return X


def run_preprocessing(input_path, output_path):
    df = pd.read_csv(input_path)

    df = df.drop(columns=["TransactionID"], errors="ignore")

    X = df.drop(columns=["IsFraud"])
    y = df["IsFraud"]

    numeric_features = ["Amount"]
    categorical_features = ["TransactionType", "Location"]

    numeric_pipeline = Pipeline(steps=[
        ("log_transform", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ("datetime", DateTimeTransformer()),
        ("preprocessor", preprocessor)
    ])

    X_processed = pipeline.fit_transform(X)

    processed_df = pd.DataFrame(X_processed)
    processed_df["IsFraud"] = y.values

    processed_df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to {output_path}")


if __name__ == "__main__":
    run_preprocessing(
        "credit_card_fraud_dataset.csv",
        "preprocessing/credit_card_fraud_preprocessed.csv"
    )


