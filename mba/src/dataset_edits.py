"""Originally dataset was created using project/create_dataset.py, we modify it there to be more interesting (since all records had collateral, bank had no way to lose money)"""

from pathlib import Path

import numpy as np
import pandas as pd

from mba.src.const import DATA_DIR, DATA_PRE_EDIT_DIR, TEST_DATA_PATH, TRAIN_DATA_PATH, TRAIN_DATA_XLSX_PATH

NON_COLLATERAL_RATE = 0.3


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    edit_data(input_path=DATA_PRE_EDIT_DIR / "loan_data.csv", output_path=TRAIN_DATA_PATH)
    edit_data(input_path=DATA_PRE_EDIT_DIR / "test_data.csv", output_path=TEST_DATA_PATH)
    # For convenience of students
    data = pd.read_csv(TRAIN_DATA_PATH)
    data.to_excel(TRAIN_DATA_XLSX_PATH)
    data = pd.read_csv(TEST_DATA_PATH)
    non_default_data = data[data["дефолт"] != 1]
    default_data = data[data["дефолт"].fillna(0) == 1]
    data = pd.concat([default_data, non_default_data.sample(frac=0.3, random_state=52)])
    data.to_csv(TEST_DATA_PATH, index=False)


def edit_data(input_path: Path, output_path: Path) -> None:
    data = pd.read_csv(input_path)
    rng = np.random.default_rng(42)
    size = data.shape[0]
    non_collateral_mask = rng.choice(size, replace=False, size=int(size * NON_COLLATERAL_RATE))
    data.loc[non_collateral_mask, "прямой_залог"] = 0
    data.loc[non_collateral_mask, "тип_залога"] = np.nan
    data.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
