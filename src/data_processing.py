import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Create aggregate transaction features per customer
    """

    def __init__(self, customer_id="CustomerId"):
        self.customer_id = customer_id

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            agg_df = X.groupby(self.customer_id).agg(
                total_amount=("Amount", "sum"),
                avg_amount=("Amount", "mean"),
                transaction_count=("TransactionId", "count"),
                std_amount=("Amount", "std")
            ).reset_index()
            return agg_df
        except Exception as e:
            raise RuntimeError(f"Aggregation failed: {e}")


class TemporalFeatures(BaseEstimator, TransformerMixin):
    """
    Extract time-based features from TransactionStartTime
    """

    def __init__(self, time_col="TransactionStartTime"):
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X = X.copy()
            X[self.time_col] = pd.to_datetime(X[self.time_col])
            X["transaction_hour"] = X[self.time_col].dt.hour
            X["transaction_day"] = X[self.time_col].dt.day
            X["transaction_month"] = X[self.time_col].dt.month
            X["transaction_year"] = X[self.time_col].dt.year
            return X
        except Exception as e:
            raise RuntimeError(f"Temporal feature extraction failed: {e}")


def build_feature_pipeline(numerical_features, categorical_features):
    """
    Build sklearn pipeline for preprocessing
    """

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def woe_placeholder():
    """
    WoE/IV transformation to be implemented using xverse or woe.
    Placeholder for regulatory-compliant encoding.
    """
    pass
