from collections.abc import Iterable
from typing import Self

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.mba.const import TARGET


def get_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data.drop(columns=TARGET), data[TARGET]


class ManualFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features: Iterable[str]) -> None:
        self.features = list(features)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features]


class ObjectToStringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()  # Avoid modifying the original DataFrame
        cat_cols = X.select_dtypes(object).columns
        X[cat_cols] = X[cat_cols].astype(str)
        return X


def get_train_test(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split(data, test_size=0.33, random_state=42)
    return df_train, df_test


def print_metrics(y_true: np.ndarray | pd.DataFrame, y_pred: np.ndarray | pd.DataFrame) -> None:
    acc = (y_pred == y_true).mean()
    print("Log reg")
    print(f"Accuracy: {acc:.3f}")
    f1 = f1_score(y_pred=y_pred, y_true=y_true)
    print(f"f1: {f1:.3f}")
