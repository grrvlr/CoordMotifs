
from typing import Optional
import numpy as np
import stumpy


def mmotifs_wrapper(X: np.ndarray, window_size: int, r: Optional[float], max_motifs: int = 5, k: int = None):

    # Discover the motifs
    mps, indices = stumpy.mstump(X.T, m=window_size)
    _, motif_set_indices, motif_set_subspaces, _ = stumpy.mmotifs(X.T, mps, indices, max_distance=r, max_motifs=max_motifs, k=k)

    # Format the motifs
    motif_sets = []
    for indices, subspace in zip(motif_set_indices, motif_set_subspaces):
        mask = np.zeros(shape=X.shape[1], dtype=bool)
        mask[subspace] = True

        motif_set = [(i, i + window_size) for i in indices if i >= 0]
        locations = np.repeat(np.array(motif_set).reshape(len(motif_set), 2, 1), subspace.shape[0], axis=-1)

        motif_sets.append((mask, locations))

    return motif_sets
