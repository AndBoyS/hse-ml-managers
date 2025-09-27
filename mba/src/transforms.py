import re
from collections.abc import Iterable
from typing import Counter, Self

import nltk
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from nltk import ngrams
from sklearn.base import BaseEstimator, TransformerMixin


def ngram_frequencies(text: str, n: int) -> pd.Series:
    # Tokenize the text into words
    tokens = nltk.word_tokenize(text)

    # Generate n-grams
    n_gram_vals = ngrams(tokens, n)

    # Count frequencies of n-grams
    n_gram_freq = Counter(n_gram_vals)
    n_gram_freq = pd.Series({" ".join(n): c for n, c in n_gram_freq.items()})

    return n_gram_freq


def remove_anomalies(data: pd.DataFrame, num_stds: float = 3) -> pd.DataFrame:
    data = data.copy()
    for col in data.columns:
        scores = scipy.stats.zscore(data[col], nan_policy="omit")
        data = data[np.abs(scores) < num_stds]

    return data


def clip_anomalies(data: pd.DataFrame, num_stds: float = 3) -> pd.DataFrame:
    data = data.copy()
    for col in data.columns:
        mean = data[col].mean()
        std = data[col].std()
        lower_val = mean - std * num_stds
        upper_val = mean + std * num_stds

        data[col] = data[col].clip(lower_val, upper_val)

    return data


class PandasTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer: BaseEstimator, col_names: list[str] | None = None) -> None:
        self.transformer = transformer
        self.col_names = col_names
        self._col_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self._col_names = self.col_names
        if self.col_names is None:
            self._col_names = list(X.columns)
        self.transformer.fit(X[self._col_names])
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        if self._col_names is None:
            raise ValueError("Pipe wasn't fitted")
        X = X.copy()
        X[self._col_names] = self.transformer.transform(X[self._col_names])
        return X


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
    def __init__(self, text_feat: str, num_ngrams: int = 50, ngrams_n: int = 3) -> None:
        self.text_feat = text_feat
        self.num_ngrams = num_ngrams
        self.ngrams_n = ngrams_n

    def _preprocess(self, t: str) -> str:
        t = re.sub(r"\W", " ", t)
        t = re.sub(r"\s+", " ", t)
        t = t.lower()
        words = t.split(" ")
        words = [w for w in words if len(w) > 2]
        words_text = " ".join(words)
        return words_text

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        col = X[self.text_feat].apply(self._preprocess)
        t = " ".join(list(col)).strip()
        self.ngrams = ngram_frequencies(t, n=self.ngrams_n)
        self.ngrams = self.ngrams.iloc[: self.num_ngrams]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        def regex_counter(s: str, ptrn: str) -> int:
            return len(re.findall(ptrn, s))

        X = X.copy()
        t_col = X[self.text_feat].apply(self._preprocess)

        for i, ngram_w in enumerate(self.ngrams.index):
            X[f"{self.text_feat}_{i}"] = t_col.str.contains(ngram_w)

        X = X.drop(columns=self.text_feat)

        # X[f"{self.text_feat}_high"] = t_col.apply(partial(regex_counter, ptrn=r"(successful|high|confident|me)"))

        return X
