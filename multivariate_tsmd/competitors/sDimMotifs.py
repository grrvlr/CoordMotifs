import stumpy
import numpy as np
import tsmd.tools.distance as distance
from joblib import Parallel, delayed

from multivariate_tsmd.competitors.competitors_tools.mStamp_tools import mdl_known_subspaces

def _fit_one_dimension(univ_signal, distance_name, wlen, distance_params):
    dist = getattr(distance, distance_name)(
        wlen, **distance_params
    )
    dist.fit(univ_signal)
    return dist

def _univariate_profile(mdim:int, distance_)->tuple:
        """Compute elementary profile of a chunk of successive lines of the crossdistance matrix

        Parameters
        ----------
        start : int
            Starting index of the chunk.
        end : int
            Ending index of the chunk (exclusive).

        Returns
        -------
        neighbors: list of np.ndarray 
            Indices of neighbors for each line in the chunk.
        dists: list of np.ndarray 
            Corresponding distances for each neighbor set.
        """
        #initialization
        D = np.empty((mdim, mdim), dtype=np.float32)
        D[0] = distance_.first_line(0)

        for i in range(1, mdim): 
            D[i] = distance_.next_line()

        return D

class sDimMotifs(object):

    def __init__(self, n_patterns, wlen, radius_ratio=3, distance_name = 'UnitEuclidean', distance_params={}, n_active_dims=None, n_jobs=1, match_type='pair'):
    
        self.n_patterns = n_patterns
        self.radius_ratio = radius_ratio
        self.distance_name = distance_name
        self.wlen=wlen
        self.n_jobs=n_jobs
        self.distance_params = distance_params
        self.n_active_dims = n_active_dims
        self.match_type = match_type

    def _multivariate_profile(self)->tuple:
        
        D = Parallel(n_jobs=self.n_jobs, prefer="processes")(
        delayed(_univariate_profile)(self.mdim_, self.distance_[d])
        for d in range(self.n_dims)
    )
        self.D = np.stack(D, axis=0)
        del D

    def search_neighbors(self, active_dims, motif_idx, pair_dist):
        neighbors = []
        sub_D = self.D[active_dims,motif_idx,:]
        mv_distances = sub_D.mean(axis=0)
        if self.radius_ratio is None:
            mv_copy = mv_distances.copy()
            mv_copy[np.isinf(mv_copy)] = np.nan
            radius = np.nanmax([np.nanmean(mv_copy) - 2*np.nanstd(mv_copy), np.nanmin(mv_copy)])
        else:
            radius = pair_dist * self.radius_ratio
        '''t_distance = pair_dist
        radius = t_distance*self.radius_ratio'''
        t_idx=np.argmin(mv_distances)

        while True:
            t_idx = np.argmin(mv_distances)
            t_distance = mv_distances[t_idx]

            if t_distance >= radius:
                break

            neighbors.append(t_idx)
            i0,j0 = max(0, t_idx - self.wlen + 1), min(self.mdim_, t_idx + self.wlen)
            mv_distances[i0:j0] = np.inf

        return neighbors
    
    def search_neighbors_pair(self, active_dims, motif_idx, nn_idx, pair_dist):
        neighbors = []
        D_motif = self.D[active_dims, motif_idx, :].mean(axis=0)
        D_nn = self.D[active_dims, nn_idx, :].mean(axis=0)
        if self.radius_ratio is None:
            D_combined = np.concatenate([D_motif, D_nn])
            D_copy = D_combined.copy()
            D_copy[np.isinf(D_copy)] = np.nan
            radius = np.nanmax([np.nanmean(D_copy) - 2*np.nanstd(D_copy), np.nanmin(D_copy)])
        else:
            radius = pair_dist * self.radius_ratio
        while True:
            # joint argmin 
            t_idx = np.argmin(np.minimum(D_motif, D_nn))
            t_distance = min(D_motif[t_idx], D_nn[t_idx])
            if t_distance >= radius:
                break

            neighbors.append(t_idx)
            i0 = max(0, t_idx - self.wlen + 1)
            i1 = min(self.mdim_, t_idx + self.wlen)
            D_motif[i0:i1] = np.inf
            D_nn[i0:i1] = np.inf
        return neighbors

    def search_next_motif(self):

        motif_set = []
        # je pense qu'on peut améliorer cette partie en faisant en sorte de ne pas retrier à chaque fois.
        perm = np.argsort(self.D, axis=0)
        D_sorted = np.take_along_axis(self.D, perm, axis=0)
        cumulated_distances = np.cumsum(D_sorted, axis=0)

        best_js = np.empty((self.n_dims, self.mdim_), dtype=int)
        best_scores = np.empty((self.n_dims, self.mdim_), dtype=float)
    
        for idx in range(self.mdim_):  
            s= cumulated_distances[:, idx, :]
            i0 = max(0, idx - self.wlen + 1)
            i1 = min(self.mdim_, idx + self.wlen)
            s[:, i0:i1] = np.inf
            best_js[:, idx] = np.argmin(s, axis=1)
            best_scores[:, idx] = s[np.arange(self.n_dims), best_js[:, idx]]
        valid_k = np.isfinite(best_scores).any(axis=1)
        valid_k_idx = np.where(valid_k)[0]
        
        if not valid_k.any():
            return None, None
        
        best_scores_v = best_scores[valid_k]
        best_js_v = best_js[valid_k]

        motifs_idxes = np.argmin(best_scores_v, axis=1)
        nn_idxes = best_js_v[np.arange(len(motifs_idxes)), motifs_idxes]

        subspaces=[]
        for i, k in enumerate(valid_k_idx):
            idx = motifs_idxes[i]
            nn = nn_idxes[i]
            dims = perm[:k+1, idx, nn]
            subspaces.append(np.sort(dims))

        mdls = mdl_known_subspaces(self.signal.T, self.wlen, motifs_idxes, nn_idxes, subspaces)
        best_i = np.argmin(mdls)
        k_real = valid_k_idx[best_i]     
        n_active_dims = k_real + 1

        active_dims = np.sort(subspaces[best_i])
        idx = motifs_idxes[best_i]
        nn_idx = nn_idxes[best_i]

        motif_set.extend([idx, nn_idx])

        pair_dist=self.D[active_dims, idx, nn_idx].mean()
        #mask
        for d in active_dims:
            for t in (idx, nn_idx):
                j0 = max(0, t - self.wlen + 1)
                j1 = min(self.mdim_, t + self.wlen)
                self.D[d, :, j0:j1] = np.inf

        if self.match_type=='single':
            neighbors= self.search_neighbors(active_dims, idx, pair_dist)
        elif self.match_type=='pair':
            neighbors = self.search_neighbors_pair(active_dims, idx, nn_idx, pair_dist)
        else:
            raise ValueError("match_type must be either 'single' or 'pair'")

        motif_set.extend(neighbors)
        for d in active_dims:
            for t in motif_set:
                i0 = max(0, t - self.wlen + 1)
                i1 = min(self.mdim_, t + self.wlen)

                self.D[d, i0:i1, :] = np.inf
                self.D[d, :, i0:i1] = np.inf
                self.mask[d, i0:i1] = True

        return motif_set, active_dims

    def fit(self,signal):
        self.signal=signal
        self.n, self.n_dims = signal.shape
        self.mdim_ = len(signal)-self.wlen+1 

        self.distance_ = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(_fit_one_dimension)(signal[:, d], self.distance_name, self.wlen, self.distance_params)
            for d in range(self.n_dims)
        )
        self._multivariate_profile()
        self.mask = np.zeros((self.n_dims, self.mdim_), dtype=bool)
        self.motifs_ = []

        for _ in range(self.n_patterns):
            if self.mask.all():
                break
            motif_set, active_dims = self.search_next_motif()
            if motif_set is None:
                break
            self.motifs_.append({
                "indices": motif_set,
                "active_dims": active_dims
            })

    @property
    def prediction_mask_(self)->np.ndarray:
        n_motifs = len(self.motifs_)
        mask = np.zeros((n_motifs, self.n), dtype=int)

        for i, motif in enumerate(self.motifs_):
            for idx in motif["indices"]:
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
            motif["active_dims"].tolist()
            for motif in self.motifs_
        ]