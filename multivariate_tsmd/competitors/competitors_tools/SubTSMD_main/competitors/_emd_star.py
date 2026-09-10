
import numpy as np
from typing import List, Tuple
from sklearn.decomposition import PCA
from competitors._lama import lama_wrapper


def emd_star_wrapper(X: np.ndarray, l_min: int, l_max: int, k_max: int, n_dims: int, elbow_deviation: float) -> List[Tuple[np.array, np.ndarray]]:
    # make the signal uni-variate by applying PCA
    pca = PCA(n_components=1)
    X_transformed = pca.fit_transform(X)

    # Apply lama to get the motif sets (as in leitmotifs)
    motif_sets_lama = lama_wrapper(
        X_transformed,
        l_min=l_min,
        l_max=l_max,
        k_max=k_max,
        n_dims=1,  # Because there is only one dimension
        elbow_deviation=elbow_deviation
    )

    # Get the relevant dimensions
    dims = np.argsort(pca.components_[:])[:, :n_dims][0]
    mask = np.zeros(shape=X.shape[1], dtype=bool)
    mask[dims] = True

    # return motif_sets_lama
    return [
        (
            mask,
            np.repeat(motif_set, X.shape[1], axis=-1)
        )
        for (_, motif_set) in motif_sets_lama
    ]
