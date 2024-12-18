from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from create_llm_feats import OUTPUT_TEST_PRIVATE, OUTPUT_TEST_PUBLIC, OUTPUT_TRAIN
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

TARGET = "дефолт"


def baseline_train(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:  # type: ignore[type-arg]
    num_feats = list(X_train.select_dtypes(exclude=object).columns)

    pipe = Pipeline(
        [
            ("feat_select", ManualFeatureSelector(num_feats)),
            ("imputer", SimpleImputer(strategy="mean")),
            (
                "classifier",
                LogisticRegression(),
            ),
        ]
    )

    return pipe.fit(X_train, y_train)


def train(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:  # type: ignore[type-arg]
    cat_feats = list(X_train.select_dtypes(object).columns)
    pipe = Pipeline(
        [
            ("object_to_string", ObjectToStringTransformer()),
            (
                "classifier",
                CatBoostClassifier(cat_features=cat_feats, allow_writing_files=False, verbose=False, random_state=3),
            ),
        ]
    )

    # imputer = IterativeImputer(max_iter=100)
    # num_cols = list(X.select_dtypes(exclude=object).columns)
    # X[num_cols] = imputer.fit_transform(X[num_cols])

    return pipe.fit(X_train, y_train)


class ManualFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):  # type: ignore[no-untyped-def]
        self.features = features

    def fit(self, X, y=None):  # type: ignore[no-untyped-def]
        return self  # No fitting necessary

    def transform(self, X):  # type: ignore[no-untyped-def]
        return X[self.features]  # Select only the specified features


class ObjectToStringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):  # type: ignore[no-untyped-def]
        return self  # No fitting necessary

    def transform(self, X):  # type: ignore[no-untyped-def]
        X = X.copy()  # Avoid modifying the original DataFrame
        cat_cols = X.select_dtypes(object).columns
        X[cat_cols] = X[cat_cols].astype(str)
        return X


def eval(model: Any, df_test_public: pd.DataFrame, df_test_private: pd.DataFrame, feats: list[str]) -> None:
    X_test_public = df_test_public[feats]
    y_test_public = df_test_public[TARGET]
    X_test_private = df_test_private[feats]
    y_test_private = df_test_private[TARGET]

    pred_public = model.predict(X_test_public)
    pred_private = model.predict(X_test_private)

    f1_public = f1_score(y_true=y_test_public, y_pred=pred_public)
    f1_private = f1_score(y_true=y_test_private, y_pred=pred_private)

    print(f"public: {f1_public}")
    print(f"private: {f1_private}")


def main() -> None:
    df_train = pd.read_csv(OUTPUT_TRAIN)
    df_test_public = pd.read_csv(OUTPUT_TEST_PUBLIC)
    df_test_private = pd.read_csv(OUTPUT_TEST_PRIVATE)

    feats = list(df_train.drop(columns=[TARGET, "сбор"]).columns)

    X_train = df_train[feats]
    y_train = df_train[TARGET]
    # model = train(X_train, y_train)
    # eval(model=model, df_test_private=df_test_private, df_test_public=df_test_public, feats=feats)

    baseline_model = baseline_train(X_train, y_train)
    eval(model=baseline_model, df_test_private=df_test_private, df_test_public=df_test_public, feats=feats)


if __name__ == "__main__":
    main()
