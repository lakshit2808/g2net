from typing import Iterable, List, Union
import pandas as pd
import functools


def merge_iters(iters: Iterable[List]) -> List:
    return functools.reduce(lambda x, y: x + y, iters)


def concat_series(serieses: Union[pd.Series, List]) -> pd.Series:
    if isinstance(serieses, pd.Series):
        serieses = serieses.tolist()
    return pd.concat(serieses, ignore_index=True)


def get_indexed_items(source_list: Iterable, indices: Iterable) -> List:
    return [source_list[id] for id in indices]
