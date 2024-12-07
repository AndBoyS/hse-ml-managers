import atexit
import functools
import json
import os
import pickle
import random
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

CUR_DIR = Path(__file__).parent
DATA_DIR = CUR_DIR / "data"

INPUT_PATH = DATA_DIR / "original/Loan_Default.csv"

KAGGLE_DIR = DATA_DIR / "kaggle"
OUTPUT_TRAIN = KAGGLE_DIR / "train.csv"
OUTPUT_TEST_TARGETS = KAGGLE_DIR / "test_targets.csv"
OUTPUT_TEST_FEAT = KAGGLE_DIR / "test_feat.csv"
OUTPUT_SUBMIT_EXAMPLE = KAGGLE_DIR / "submission_example.csv"
CACHE_PATH_ESSAY = DATA_DIR / "llm_essay_feats.pkl"
SYNONYMS_PATH = DATA_DIR / "word_synonyms.json"
MAX_REQUEST_TRIES = 10

PROMPT_TEMPLATE_ESSAY = """You will be given a list of bank clients' data with different attributes.
For each client, write a 50-word text from client's point of view, as if he is introducing himself to a bank employee.
Some info and instructions you need to keep in mind:
- if "income" attribute equals medium, that means client has a stable job, if income is high, then client is rich
- "approv_in_adv" attribute equals 1 if loan for client approved in advanced, otherwise 0.
- if attribute "Status" equals 1, it means that the client will not be able to pay back the loan, if attribute "Status" equals 0, then client will repay the loan
- come up with personal details, if attribute "Status" equals 1, add subtle hints that he has problems with money/life, or add some confusion to the narration.
- if attribute "Status" equals 0, dont add subtle hints that he has problems with money/life, and dont add any confusion to the narration.
- add some variation to text structure between each client, use different figures of speech

Format of your answer is:
"<speech of customer 1>"
"<speech of customer 2>"
...

Here's the clients' info:

{info}
"""


CachedObj = dict[str, str]


class Cache:
    def __init__(self, path: Path) -> None:
        self.cache = self.get_cache(path)

    @staticmethod
    @functools.cache  # Для идентичности объектов кэша при повторном вызове
    def get_cache(path: Path) -> CachedObj:
        """
        Получить dict, который будет кэшироваться в path
        В течение работы программы данные хранятся в ОЗУ: выгрузка осуществляется при завершении программы
        """

        def load() -> CachedObj:
            with path.open("rb") as file:
                try:
                    res: CachedObj = pickle.load(file)
                    return res
                except EOFError:
                    return {}

        def dump(cache: CachedObj) -> None:
            with path.open("wb") as file:
                pickle.dump(cache, file)

        path.touch(exist_ok=True)
        cache = load()
        atexit.register(lambda: dump(cache))

        return cache

    def add_new_el(self, k: str, v: str) -> None:
        self.cache[k] = v

    def reset_cache(self) -> None:
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)


def random_nan_placement(df: pd.DataFrame, rate: float, seed: int = 1, exclude: list[str] | None = None) -> None:
    np.random.seed(seed)
    num_cols = list(df.select_dtypes(exclude=object).columns)

    if exclude:
        for el in exclude:
            num_cols.remove(el)

    idx: list[tuple[int, str]] = []
    for col in num_cols:
        for i in df[col].dropna().index:
            idx.append((i, col))

    num_nans = int(len(idx) * rate)

    idx_cont = range(len(idx))
    nan_idx_cont = np.random.choice(idx_cont, replace=False, size=(num_nans,))
    nan_idx = [idx[i] for i in nan_idx_cont]

    for i in nan_idx:
        df.loc[*i] = np.nan


def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "loan_limit": "лимит_нарушен",
            "Gender": "пол",
            "loan_type": "тип",
            "loan_purpose": "цель",
            "Credit_Worthiness": "кредитоспособность",
            "open_credit": "другие_кредиты",
            "business_or_commercial": "бизнес",
            "loan_amount": "сумма",
            "Upfront_charges": "сбор",
            "term": "срок",
            "Neg_ammortization": "амортизация",
            "interest_only": "только_процент",
            "lump_sum_payment": "один_платеж",
            "property_value": "стоимость_имущества",
            "occupancy_type": "работа",
            "Secured_by": "тип_залога",
            "credit_type": "тип_кредита",
            "Credit_Score": "кредитный_рейтинг",
            "age": "возраст",
            "Security_Type": "прямой_залог",
            "Status": "дефолт",
            "essay": "речь",
        }
    )


def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    COLS_TO_REMOVE = [
        "rate_of_interest",
        "Interest_rate_spread",
        "approv_in_adv",
        "Region",
        "total_units",
        "income",
        "LTV",
        "dtir1",
        "year",
        "construction_type",
        "co-applicant_credit_type",
        "submission_of_application",
    ]
    return df.drop(columns=COLS_TO_REMOVE)


def preprocess() -> pd.DataFrame:
    NUM_ROWS = 10000
    df = pd.read_csv(INPUT_PATH)
    df = df.sample(NUM_ROWS, random_state=42, replace=False)
    rd = random.Random()
    rd.seed(42)
    df["ID"] = [str(uuid.UUID(int=rd.getrandbits(128), version=4)) for _ in range(df.shape[0])]
    df = df.set_index("ID")

    income_mapping = {0.33: "low", 0.66: "medium", 1.0: "high"}
    df["income"] = map_floats_to_words(df["income"], quantile_mapping=income_mapping)

    # Upfront charges
    upfront_mapping = {0: "none", 0.33: "small", 0.66: "medium", 1.0: "high"}
    df["Upfront_charges"] = map_floats_to_words(df["Upfront_charges"], quantile_mapping=upfront_mapping)

    with SYNONYMS_PATH.open() as f:
        synonyms_dict: dict[str, list[str]] = json.load(f)
    new_vals: list[str] = []
    rng = random.Random(42)
    for val in df["Upfront_charges"]:
        if not isinstance(val, str):
            val = "none"
        new_vals.append(rng.choice(synonyms_dict[val]))
    df["Upfront_charges"] = new_vals

    num_cols = df.select_dtypes(exclude=object).columns
    stds = df[num_cols].std()
    for col in num_cols:
        if col == "Status":
            continue
        df[col] += stds[col] * 0.1

    df["occupancy_type"] = df["occupancy_type"].map({"pr": "осн", "sr": "втор", "ir": "инвест"})
    df["Secured_by"] = df["Secured_by"].map({"land": "земля", "home": "дом"})
    df["credit_type"] = df["credit_type"].map({"CIB": 1, "CRIF": 2, "EXP": 3, "EQUI": 4})
    df["loan_limit"] = df["loan_limit"] == "ncf"
    df["loan_purpose"] = 4 - df["loan_purpose"].str[1].astype(float)
    df["loan_type"] = 4 - df["loan_type"].str[-1].astype(float)
    df["Credit_Worthiness"] = 2 - df["Credit_Worthiness"].str[1].astype(float)
    df["open_credit"] = df["open_credit"] == "opc"
    df["business_or_commercial"] = df["business_or_commercial"] == "b/c"
    df["Neg_ammortization"] = df["Neg_ammortization"] == "neg_amm"
    df["interest_only"] = df["interest_only"] == "int_only"
    df["lump_sum_payment"] = df["lump_sum_payment"] == "lpsm"
    df["Security_Type"] = df["Security_Type"] == "direct"
    df["Gender"] = df["Gender"].map({"Female": "ж", "Male": "м", "Sex Not Available": "n/a"})

    bool_cols = df.select_dtypes(bool).columns
    df[bool_cols] = df[bool_cols].astype(float)

    random_nan_placement(df, rate=0.2, seed=4, exclude=["Status"])
    return df


def map_floats_to_words(x: pd.Series, quantile_mapping: dict[float, str]) -> pd.Series:  # type: ignore[type-arg]
    def map_quantile(value: float, threshs: Iterable[float], names: Iterable[str]) -> str | float:
        if np.isnan(value):
            return value
        for thresh, name in zip(threshs, names):
            if value <= thresh:
                return name
        raise ValueError

    quantiles = list(quantile_mapping)
    names = list(quantile_mapping.values())
    threshs = x.quantile(quantiles)
    return x.apply(functools.partial(map_quantile, threshs=threshs, names=names))


def create_essay_prompt(df: pd.DataFrame) -> str:
    info = "\n".join([row.to_json() for _, row in df.iterrows()])
    return PROMPT_TEMPLATE_ESSAY.replace("{info}", str(info))


def iter_batch(df: pd.DataFrame, batch_size: int = 10) -> Iterator[pd.DataFrame]:
    for i in range(0, df.shape[0], batch_size):
        yield df.iloc[i : i + batch_size]


def ask_question(prompt: str) -> list[str]:
    url = "https://api.proxyapi.ru/openai/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return [d["message"]["content"] for d in response.json()["choices"]]

    raise requests.RequestException(response.status_code, response.text)


def try_ask_question_and_save(batch: pd.DataFrame, prompt: str, cache: Cache) -> bool:
    resp = ask_question(prompt)[0]
    return save_response(df=batch, resp=resp, cache=cache)


def save_response(df: pd.DataFrame, resp: str, cache: Cache) -> bool:
    resp_list = [line for line in resp.splitlines() if line.startswith('"') and line.endswith("")]
    if len(resp_list) != df.shape[0]:
        return False
    for i, line in zip(df.index, resp_list):
        print(line)
        cache.add_new_el(k=i, v=line)
    return True


def collect_essays(df: pd.DataFrame, cache: Cache) -> None:
    not_done_idx = [i for i in df.index if i not in cache.cache]
    df = df.loc[not_done_idx]

    for batch in tqdm(list(iter_batch(df))):
        prompt = create_essay_prompt(batch)
        batch_done = False

        for _ in range(MAX_REQUEST_TRIES):
            try:
                if try_ask_question_and_save(batch=batch, prompt=prompt, cache=cache):
                    batch_done = True
                    break
            except requests.RequestException as e:
                print(e)

        if not batch_done:
            print("Skipped batch")


def add_essays_to_df(df: pd.DataFrame, cache: Cache) -> None:
    essay_col = "essay"
    df[essay_col] = np.nan
    df[essay_col] = df[essay_col].astype(object)

    for i, line in cache.cache.items():
        df.loc[i, essay_col] = line


def main() -> None:
    df = preprocess()
    cache = Cache(CACHE_PATH_ESSAY)
    collect_essays(df=df, cache=cache)
    add_essays_to_df(df=df, cache=cache)
    df = rename_cols(df)
    df = drop_cols(df)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_test_public, df_test_private = train_test_split(df_test, test_size=0.5, random_state=42)

    OUTPUT_TRAIN.parent.mkdir(exist_ok=True)
    df_train.to_csv(OUTPUT_TRAIN)

    df_test = pd.concat((df_test_public, df_test_private))
    df_test.drop(columns="дефолт").to_csv(OUTPUT_TEST_FEAT)

    df_test_public["Usage"] = "Public"
    df_test_private["Usage"] = "Private"
    df_target = pd.concat((df_test_public, df_test_private))[["дефолт", "Usage"]]
    df_target.to_csv(OUTPUT_TEST_TARGETS)

    submit = df_target["дефолт"].copy()
    submit[:] = np.random.randint(0, 2, size=submit.shape[0])  # type: ignore[call-overload]
    submit.to_csv(OUTPUT_SUBMIT_EXAMPLE)


if __name__ == "__main__":
    main()
