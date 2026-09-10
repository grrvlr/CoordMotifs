# STUMPY
# Copyright 2019 TD Ameritrade. Released under the terms of the 3-Clause BSD license.
# STUMPY is a trademark of TD Ameritrade IP Company, Inc. All rights reserved.

import warnings
import stumpy
import numpy as np

from multivariate_tsmd.competitors.competitors_tools import config, mStamp_tools as core 

class mMotifs(object):

    def __init__(self, wlen, n_patterns=None, n_dims=None, excl_zone_denom=None):

        self.wlen=wlen
        self.n_patterns=n_patterns
        self.n_dims=n_dims
        self.excl_zone_denom = excl_zone_denom
        if self.n_patterns is not None:
            self.cutoffs = np.inf
        else:
            self.cutoffs = None

    def fit(self,signal):
        self.signal=signal.T
        self.n=self.signal.shape[1]
        self.mps,self.indices=stumpy.mstump(self.signal,self.wlen)
        self.distances, self.motifs_idx, self.motifs_dims, self.motifs_mdls = core.mmotifs(
            self.signal,
            self.mps,
            self.indices, max_motifs=self.n_patterns, cutoffs=self.cutoffs, excl_zone_denom=self.excl_zone_denom
        )

    @property
    def prediction_mask_(self)->np.ndarray:
        n_motifs = len(self.motifs_idx)

        mask = np.zeros((n_motifs, self.n), dtype=int)
        for i, motif in enumerate(self.motifs_idx):
            for idx in motif:
                if idx == -1 :
                    break
                start = idx
                end = min(idx + self.wlen, self.n)
                mask[i, start:end] = 1

        return mask

    @property
    def prediction_dimension_(self) -> list:
        """
        Active dimensions for each detected motif.
        """
        return [
            active_dims.tolist()
            for active_dims in self.motifs_dims
        ]

    
