from pathlib import Path

REPO_DIR = Path(__file__).parents[2]

DATA_PRE_EDIT_DIR = REPO_DIR / "mba/data_pre_edit"
DATA_DIR = REPO_DIR / "mba/data"
TRAIN_DATA_PATH = DATA_DIR / "loan_data.csv"
TRAIN_DATA_XLSX_PATH = TRAIN_DATA_PATH.with_suffix(".xlsx")
TEST_DATA_PATH = DATA_DIR / "test_data.csv"

TARGET = "дефолт"
RATE = 0.1
NUM_MONTHS = 12
