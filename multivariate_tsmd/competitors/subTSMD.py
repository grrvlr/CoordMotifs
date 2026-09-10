import os
import pandas as pd
import numpy as np 
from competitors_tools.SubTSMD_main.sub_tsmd_tools._SubTSMD import SubTSMD as SubTSMD_original

class SubTSMD:
    def __init__(self, min_wlen, max_wlen, n_patterns=None, rho=0.65):
        self.min_wlen = min_wlen
        self.max_wlen = max_wlen
        self.n_patterns = n_patterns
        self.rho = rho

    def fit(self, signal):
        self.n, self.n_dims =signal.shape
        subspace_motif_discovery = SubTSMD_original(
        l_min=self.min_wlen,
        l_max=self.max_wlen,
        rho=self.rho,
        max_number_motif_sets=self.n_patterns,
        linkage='average',
    )
        self.subspace_motifs = subspace_motif_discovery.apply(signal)
        self.predicted_n_motifs = len(self.subspace_motifs)

    @property
    def prediction_mask_(self):
        mask = np.zeros((self.predicted_n_motifs, self.n), dtype=int)
        for i, motif_set in enumerate(self.subspace_motifs):
            motif_set=motif_set.to_motifs
            for motif in motif_set:
                indices = motif._indices
                s = int(np.min(indices[0,:]))
                e = int(np.max(indices[1,:]))
                mask[i, s:e] = 1

        return mask
    
    @property
    def prediction_dimension_(self):
        dims = []
        for i, motif_set in enumerate(self.subspace_motifs):
            motif_set=motif_set.to_motifs
            dims.append(np.unique(motif_set[0].subspace).tolist())
        return dims

    def get_subspace_motifs(self):
        return self.subspace_motifs