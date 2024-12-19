from typing import cast

import numpy as np
import pandas as pd

from src.mba.const import NUM_MONTHS, RATE

arr = np.ndarray | pd.Series


def calculate_interest(start: arr, num_periods: arr, rate: float) -> arr:
    return start * (1 + rate) ** num_periods - start


def profit(y_true: arr, y_pred: arr, feats: pd.DataFrame) -> float:
    feats = feats.copy()
    success_mask = (y_true == 0) & (y_pred == 0)
    start_sum = feats.loc[success_mask, "сумма"]
    num_periods = feats.loc[success_mask, "срок"] / NUM_MONTHS
    profit = calculate_interest(start=start_sum, num_periods=num_periods, rate=RATE).sum()

    failure_mask = (y_true == 1) & (y_pred == 0)

    loss = feats.loc[failure_mask, "сумма"].fillna(0)
    # Используем залог
    loss -= feats.loc[failure_mask, "стоимость_имущества"].fillna(0)
    loss = loss.clip(lower=0)
    profit -= loss
    return cast(float, profit.sum())
