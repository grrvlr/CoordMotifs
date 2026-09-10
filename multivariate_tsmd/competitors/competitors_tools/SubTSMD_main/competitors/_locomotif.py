
from typing import List, Tuple
import numpy as np
from locomotif import locomotif


def z_normalize(X: np.array) -> np.array:
    return (X - X.mean()) / X.std()


def univariate_locomotif_wrapper(X: np.ndarray, l_min: int, l_max: int, rho: float, warping: bool, nb: int = None, start_mask=None, end_mask=None) -> List[List[Tuple[np.array, np.ndarray]]]:

    dimension = X.shape[1]
    univariate_motif_sets = []

    for d in range(dimension):
        mask = np.zeros(shape=dimension, dtype=bool)
        mask[d] = True

        motif_sets = locomotif.apply_locomotif(z_normalize(X[:, d]), rho=rho, l_min=l_min, l_max=l_max, nb=nb, warping=warping, overlap=0.0, start_mask=start_mask, end_mask=end_mask)
        univariate_motif_sets.append([
            (mask, np.expand_dims(np.array(motif_set), axis=-1))
            for (_, motif_set) in motif_sets
        ])

    return univariate_motif_sets
