import asyncio
import atexit
import functools
import pickle
import random
import uuid
from collections.abc import Iterator
from pathlib import Path

import duck_chat
import pandas as pd
from aiohttp import client_exceptions
from duck_chat.exceptions import DuckChatException
from tqdm import tqdm

CACHE_PATH = Path(__file__).parent / "default_llm_essay_feats.pkl"
MAX_REQUEST_TRIES = 10

PROMPT_TEMPLATE = """You will be given a list of bank clients' data with different attributes.
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


def preprocess() -> pd.DataFrame:
    df_path = Path(__file__).parent / "data/Loan_Default.csv"
    NUM_ROWS = 10000
    COLS_TO_REMOVE = ["rate_of_interest", "Interest_rate_spread"]
    df = pd.read_csv(df_path)
    df = df.drop(columns=COLS_TO_REMOVE)
    df = df.sample(NUM_ROWS, random_state=42, replace=False)
    rd = random.Random()
    rd.seed(42)
    df["ID"] = [str(uuid.UUID(int=rd.getrandbits(128), version=4)) for _ in range(df.shape[0])]
    return df


def preprocess_for_prompt(df: pd.DataFrame) -> pd.DataFrame:
    def map_quantile(value: float, mapping: dict[float, float]):
        if value <= mapping[0.33]:
            return "low"
        elif value <= mapping[0.66]:
            return "medium"
        else:
            return "high"

    def categorize_income(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        quantiles_to_use = [0.33, 0.66, 1.0]
        quantiles = df["income"].quantile(quantiles_to_use)
        mapping = {k: v for k, v in zip(quantiles_to_use, quantiles)}
        df["income"] = df["income"].apply(functools.partial(map_quantile, mapping=mapping))
        return df

    df = df.set_index("ID")
    COLS_TO_USE = [
        "approv_in_adv",
        "Gender",
        "income",
        "Status",
    ]
    df = categorize_income(df)
    return df[COLS_TO_USE]


def create_prompt(df: pd.DataFrame) -> str:
    info = "\n".join([row.to_json() for _, row in df.iterrows()])
    return PROMPT_TEMPLATE.replace("{info}", str(info))


def iter_batch(df: pd.DataFrame, batch_size: int = 10) -> Iterator[pd.DataFrame]:
    for i in range(0, df.shape[0], batch_size):
        yield df.iloc[i : i + batch_size]


def save_response(df: pd.DataFrame, resp: str, cache: Cache) -> None:
    resp_list = [line for line in resp.splitlines() if line.startswith('"') and line.endswith("")]
    if len(resp_list) != df.shape[0]:
        raise ValueError

    for i, line in zip(df.index, resp_list):
        cache.add_new_el(k=i, v=line)


async def main():
    df = preprocess()
    df.to_csv(Path(__file__).parent / "incomplete_data.csv", index=False)
    df = preprocess_for_prompt(df)

    cache = Cache(CACHE_PATH)

    not_done_idx = [i for i in df.index if i not in cache.cache]
    df = df.loc[not_done_idx]

    for batch in tqdm(list(iter_batch(df))):
        async with duck_chat.DuckChat(model=duck_chat.ModelType.GPT4o) as chat:
            prompt = create_prompt(batch)
            batch_done = False
            for _ in range(MAX_REQUEST_TRIES):
                try:
                    resp = await chat.ask_question(prompt)
                    save_response(df=batch, resp=resp, cache=cache)
                except (
                    ValueError,
                    DuckChatException,
                    client_exceptions.ClientPayloadError,
                    client_exceptions.ServerDisconnectedError,
                ) as e:
                    print(e)
                    continue
                else:
                    batch_done = True
                    break

            if not batch_done:
                print("Skipped batch")


if __name__ == "__main__":
    asyncio.run(main())
