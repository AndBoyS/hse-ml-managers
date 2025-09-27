from collections.abc import Iterable

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

from .const import TARGET


def plot_hist(data: pd.DataFrame, cols: list[str], bins: int = 60) -> None:
    data[cols].hist(bins=bins, figsize=(7, 7))
    plt.suptitle("Распределение числовых признаков")
    plt.show()


def plot_corr(data: pd.DataFrame, cols: list[str]) -> None:
    corr = data[cols].corr()
    mask = np.triu(corr).astype(bool)
    sns.heatmap(corr, annot=True, mask=mask)
    plt.title("Корреляционная матрица")
    plt.show()


def plot_scatter(data: pd.DataFrame, cols: list[str]) -> None:
    g = sns.PairGrid(data, x_vars=cols, y_vars=cols)
    g.map_lower(sns.scatterplot)
    plt.suptitle("Точечные графики числовых признаков")
    plt.tight_layout()
    plt.show()


def plot_nan(data: pd.DataFrame) -> None:
    msno.matrix(data)


def print_value_counts(data: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        print(data[col].value_counts())
        print()


def plot_categorical_distributions(data: pd.DataFrame, cols: list[str]) -> None:
    n_plot_cols = 2
    n_plot_rows = (len(cols) + n_plot_cols - 1) // n_plot_cols

    plt.figure(figsize=(12, n_plot_rows * 4))

    for i, col in enumerate(cols):
        plt.subplot(n_plot_rows, n_plot_cols, i + 1)
        sns.countplot(data=data, x=col)
        plt.title(f"Распределение {col}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def visualize_cat_and_target(data: pd.DataFrame, cols: list[str]) -> None:
    n_plot_cols = 2
    n_plot_rows = (len(cols) + n_plot_cols - 1) // n_plot_cols

    plt.figure(figsize=(12, n_plot_rows * 4))
    for i, col in enumerate(cols):
        plt.subplot(n_plot_rows, n_plot_cols, i + 1)
        # fig, ax = plt.subplots(figsize=(3, 2))
        sns.countplot(x=col, hue=TARGET, data=data)
        plt.title(f"Распределение таргета для признака {col}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_anomalies(data: pd.DataFrame, cols: list[str], n_bins: int = 100) -> None:
    for col in cols:
        col_data = data[col].dropna()
        mean = col_data.mean()
        std_dev = col_data.std()

        lower_threshold = mean - 3 * std_dev
        upper_threshold = mean + 3 * std_dev

        counts, bins = np.histogram(col_data, bins=n_bins)
        colors = [
            "blue" if (lower_threshold <= (bin_start + bin_end) / 2 <= upper_threshold) else "red"
            for bin_start, bin_end in zip(bins[:-1], bins[1:])
        ]

        plt.figure(figsize=(10, 6))
        plt.bar(bins[:-1], counts, width=np.diff(bins), color=colors, edgecolor="black", align="edge")
        plt.axvline(mean, color="green", linestyle="dashed", linewidth=1, label="Mean")
        plt.axvline(lower_threshold, color="orange", linestyle="dashed", linewidth=1, label="-3 Std Dev")
        plt.axvline(upper_threshold, color="orange", linestyle="dashed", linewidth=1, label="+3 Std Dev")

        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()
