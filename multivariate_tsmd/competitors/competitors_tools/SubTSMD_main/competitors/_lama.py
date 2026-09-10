
import numpy as np
from typing import List, Tuple

from leitmotifs.plotting import LAMA


def lama_wrapper(X: np.ndarray, l_min: int, l_max: int, k_max: int, n_dims: int, elbow_deviation: float) -> List[Tuple[np.array, np.ndarray]]:

    # Initialize the LAMA object
    lama = LAMA(
        ds_name="ONLY-USED-FOR-PLOTTING",
        series=X.T,
        n_dims=n_dims,
        slack=elbow_deviation
    )

    # Compute the motif length
    if l_min == l_max:
        motif_length, _ = lama.fit_motif_length(
            k_max=k_max,
            motif_length_range=np.arange(l_min, l_max + 1, 1),
            plot=False,
            plot_elbows=False,
            plot_motifsets=False,
            plot_best_only=False,
        )
    else:
        motif_length = l_min

    # Compute the motifs
    _, motif_sets_lama, elbow_points = lama.fit_k_elbow(
        k_max=k_max,
        motif_length=motif_length,
        plot_elbows=False,
        plot_motifsets=False,
    )

    # Format the motif sets
    motif_sets = []
    for elbow in elbow_points:
        mask = np.zeros(shape=X.shape[1], dtype=bool)
        mask[lama.leitmotifs_dims[elbow]] = True

        motif_set = [(s, s + motif_length) for s in motif_sets_lama[elbow]]
        locations = np.repeat(np.array(motif_set).reshape(len(motif_set), 2, 1), lama.leitmotifs_dims[elbow].shape[0], axis=-1)

        motif_sets.append((mask, locations))

    return motif_sets
