from typing import cast

import numpy as np
import pandas as pd

from .const import RATE

arr = np.ndarray | pd.Series


def calculate_interest(start: arr, num_periods: arr, rate: float) -> arr:
    """
    Calculate total interest income for bank from annuity loan
    Parameters:
    start (array) - loan amounts
    num_periods (array) - number of months for each loan
    rate (float) - annual interest rate (e.g. 0.12 for 12%)
    Returns:
    array - total interest income for each loan
    """
    # Monthly rate
    monthly_rate = rate / 12

    # Calculate monthly payment using annuity formula
    monthly_payment = (
        start * (monthly_rate * (1 + monthly_rate) ** num_periods) / ((1 + monthly_rate) ** num_periods - 1)
    )

    # Total amount paid = monthly payment * number of periods
    total_paid = monthly_payment * num_periods

    # Interest is the difference between total paid and initial loan
    return total_paid - start


def profit(y_true: arr, y_pred: arr, feats: pd.DataFrame) -> float:
    feats = feats.copy()
    success_mask = (y_true == 0) & (y_pred == 0)
    start_sum = feats[success_mask.to_list()]["сумма"].fillna(0)
    num_periods = feats[success_mask.to_list()]["срок"].fillna(365) // 30
    profit = calculate_interest(start=start_sum, num_periods=num_periods, rate=RATE).sum()

    failure_mask = (y_true == 1) & (y_pred == 0)

    loss = feats[failure_mask.to_list()]["сумма"].fillna(0)
    loss -= feats[failure_mask.to_list()]["стоимость_имущества"].fillna(0)
    profit -= loss.sum()
    return cast(float, profit)
