import itertools
import re
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.mba.const import TARGET


def get_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data.drop(columns=TARGET), data[TARGET]


class DataFramer(BaseEstimator, TransformerMixin):
    def __init__(self, features: Iterable[str]) -> None:
        self.features = list(features)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        return self

    def transform(self, X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X, columns=self.features)


class ManualFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features: Iterable[str]) -> None:
        self.features = list(features)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features]


class TextFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, text_feat: str) -> None:
        self.text_feat = text_feat

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        def regex_counter(s: str, ptrn: str) -> int:
            return len(re.findall(ptrn, s))

        text = X[self.text_feat]
        X[f"{self.text_feat}_exc"] = text.str.count("!")
        X[f"{self.text_feat}_high"] = text.apply(
            partial(regex_counter, ptrn=r"(?<!not\s)(high|substantial|best|prosper|wealth|great|successful)")
        )
        X[f"{self.text_feat}_medium"] = text.apply(
            partial(regex_counter, ptrn=r"(?<!not\s)(medium|solid|confiden|\sreliable|\sbalance|excit)")
        )
        X[f"{self.text_feat}_modest"] = text.apply(partial(regex_counter, ptrn=r"(modest)"))
        X[f"{self.text_feat}_stable"] = text.apply(
            partial(regex_counter, ptrn=r"(?<!not\s)(planning|stability|\sstable|commited)")
        )
        X[f"{self.text_feat}_meh"] = text.apply(
            partial(regex_counter, ptrn=r"(rocky|unsure|rough|struggl|low|not sure)")
        )
        return X


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


def grid_search(params_dict: dict[str, Any]) -> Iterator[dict[str, Any]]:
    param_combs = itertools.product(*list(params_dict.values()))
    param_names = list(params_dict)

    for param_vals in param_combs:
        config_kwargs: dict[str, Any] = {}

        for param_name, param_val in zip(param_names, param_vals):
            config_kwargs[param_name] = param_val

        yield config_kwargs
