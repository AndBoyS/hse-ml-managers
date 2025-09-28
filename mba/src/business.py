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


def calculate_profit(y_true: arr, y_pred: arr, feats: pd.DataFrame) -> float:
    feats = feats.copy()
    success_mask = (y_true == 0) & (y_pred == 0)
    failure_mask = (y_true == 1) & (y_pred == 0)

    start_sums = feats["сумма"].fillna(0)
    num_periods = feats["срок"].fillna(0)
    potential_profits = calculate_interest(start=start_sums, num_periods=num_periods, rate=RATE)

    profit = potential_profits[success_mask].sum()

    losses = start_sums[failure_mask]
    collateral = feats["стоимость_имущества"].fillna(0)
    collateral[feats["прямой_залог"] != 1] = 0
    losses -= collateral[failure_mask]
    losses = losses.clip(lower=0)

    profit -= losses.sum()
    return cast(float, profit)
