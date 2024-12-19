import re
from collections.abc import Iterable
from functools import partial
from typing import Self

import numpy as np
import pandas as pd
import scipy
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin


def remove_anomalies(data: pd.DataFrame, cols: list[str], thresh: int = 3) -> pd.DataFrame:
    data = data.copy()
    for col in cols:
        scores = scipy.stats.zscore(data[col], nan_policy="omit")
        data = data[(np.abs(scores) < thresh)]

    return data


def clip_anomalies(data: pd.DataFrame, cols: list[str], thresh: int = 3) -> pd.DataFrame:
    data = data.copy()
    for col in cols:
        mean = data[col].mean()
        std = data[col].std()
        lower_val = mean - std * thresh
        upper_val = mean + std * thresh

        data[col] = data[col].clip(lower_val, upper_val)

    return data


class DataFramer(BaseEstimator, TransformerMixin):
    def __init__(self, features: Iterable[str]) -> None:
        self.features = list(features)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Self:
        return self

    def transform(self, X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(X, columns=self.features)


class FeatSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features: Iterable[str]) -> None:
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[list(self.features)]


class TextFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, text_feat: str) -> None:
        self.text_feat = text_feat

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        def regex_counter(s: str, ptrn: str) -> int:
            return len(re.findall(ptrn, s))

        X = X.copy()

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
        X = X.drop(columns=self.text_feat)
        return X


class ObjectToStringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cat_cols = X.select_dtypes(object).columns
        X[cat_cols] = X[cat_cols].astype(str)
        return X
