import pickle

import numpy as np


def _load(
    path: str,
) -> (list[np.ndarray], list[list[(np.array, list[(np.array, np.ndarray)])]]):
    with open(path, "rb") as file:
        df = pickle.load(file)
    return df["ts"].to_list(), df["gt"].to_list()


def load_validation(
    path: str,
) -> (list[np.ndarray], list[list[(np.array, list[(np.array, np.ndarray)])]]):
    return _load(f"{path}/validation.pkl")


def load_test(
    path: str,
) -> (list[np.ndarray], list[list[(np.array, list[(np.array, np.ndarray)])]]):
    return _load(f"{path}/test.pkl")


def load(path: str) -> (np.ndarray, list[(np.array, list[(np.array, np.ndarray)])]):
    ts, gt = _load(path)
    assert (
        len(ts) == 1
    ), "The 'load' method should only be used for loading single time series!"
    return ts[0], gt[0]
