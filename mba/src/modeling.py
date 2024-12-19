import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from .const import TARGET


def get_xy(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data.drop(columns=TARGET), data[TARGET]


def get_train_test(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split(data, test_size=0.33, random_state=42)
    return df_train, df_test


def print_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> None:
    acc = (y_pred == y_true).mean()
    print("Log reg")
    print(f"Accuracy: {acc:.3f}")
    f1 = f1_score(y_pred=y_pred, y_true=y_true)
    print(f"f1: {f1:.3f}")
