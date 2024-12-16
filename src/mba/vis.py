from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest

from src.mba.const import TARGET


def get_float_cols_for_vis(loan_data: pd.DataFrame) -> list[str]:
    cols = loan_data.select_dtypes(exclude=object).columns
    return [c for c in cols if loan_data[c].nunique() >= 20]


def get_cat_cols_for_vis(loan_data: pd.DataFrame, for_plot: bool = False) -> list[str]:
    float_cols = get_float_cols_for_vis(loan_data)
    cat_cols = list(set(loan_data.columns) - set(float_cols))
    if for_plot:
        cat_cols = [c for c in cat_cols if loan_data[c].nunique() < 20]
    return cat_cols


def plot_hist(loan_data: pd.DataFrame) -> None:
    float_cols = get_float_cols_for_vis(loan_data)
    loan_data[float_cols].hist(bins=60, figsize=(7, 7))
    plt.suptitle("Распределение числовых признаков")
    plt.show()


def plot_scatter(loan_data: pd.DataFrame) -> None:
    float_cols = get_float_cols_for_vis(loan_data)
    g = sns.PairGrid(loan_data, x_vars=float_cols, y_vars=float_cols)
    g.map_lower(sns.scatterplot)
    plt.suptitle("Точечные графики числовых признаков")
    plt.tight_layout()
    plt.show()


def print_value_counts(loan_data: pd.DataFrame) -> None:
    cat_cols = get_cat_cols_for_vis(loan_data)
    for col in cat_cols:
        print(loan_data[col].value_counts())
        print()


def plot_categorical_distributions(loan_data: pd.DataFrame) -> None:
    cat_feats = get_cat_cols_for_vis(loan_data, for_plot=True)

    num_cols = 2
    num_rows = (len(cat_feats) + num_cols - 1) // num_cols

    plt.figure(figsize=(12, num_rows * 4))

    for i, col in enumerate(cat_feats):
        plt.subplot(num_rows, num_cols, i + 1)
        sns.countplot(data=loan_data, x=col)
        plt.title(f"Распределение {col}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def visualize_cat_and_target(loan_data: pd.DataFrame) -> None:
    cat_feats = get_cat_cols_for_vis(loan_data, for_plot=True)
    num_cols = 2
    num_rows = (len(cat_feats) + num_cols - 1) // num_cols

    plt.figure(figsize=(12, num_rows * 4))
    for i, col in enumerate(cat_feats):
        plt.subplot(num_rows, num_cols, i + 1)
        # fig, ax = plt.subplots(figsize=(3, 2))
        sns.countplot(x=col, hue=TARGET, data=loan_data)
        plt.title(f"Распределение таргета для признака {col}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def get_anomaly_mask(loan_data: pd.DataFrame, col: str) -> np.ndarray:
    model = IsolationForest(contamination=0.05)
    return cast(np.ndarray, model.fit_predict(loan_data[[col]]) == -1)


def plot_anomalies(loan_data: pd.DataFrame, bins: int = 100) -> None:
    float_cols = get_float_cols_for_vis(loan_data)
    for col in float_cols:
        anomaly_mask = get_anomaly_mask(loan_data, col)

        plt.figure(figsize=(10, 6))
        plt.hist(loan_data[col][~anomaly_mask], bins=bins, alpha=0.5, label="Normal", color="blue")
        plt.hist(loan_data[col][anomaly_mask], bins=bins, alpha=0.5, label="Anomaly", color="red")
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
