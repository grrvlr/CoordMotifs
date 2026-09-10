from leitmotifs.plotting import LAMA as original_LAMA
import numpy as np

class LAMA(object):

    def __init__(self, k_max, min_wlen, max_wlen, n_dims=None, distance="znormed_ed", elbow_deviation=1.00, slack=1, n_jobs=1, plot = False) -> None:

        self.k_max=k_max
        self.min_wlen=min_wlen
        self.max_wlen=max_wlen
        self.n_dims=n_dims 
        self.distance=distance
        self.elbow_deviation=elbow_deviation
        self.slack=slack
        self.n_jobs=n_jobs
        self.plot=plot

    def fit(self, signal):
        self.signal=signal 
        self.n,self.d=self.signal.shape
        self.lama=original_LAMA(ds_name='signal',series=self.signal.T, n_dims=self.n_dims, distance=self.distance, elbow_deviation=self.elbow_deviation, slack=self.slack, n_jobs=self.n_jobs)
        self.motif_length,_=self.lama.fit_motif_length(k_max=self.k_max, motif_length_range=[self.min_wlen,self.max_wlen],plot=self.plot, plot_motifsets=self.plot, plot_best_only=self.plot)
        _,self.leitmotifs,self.elbow_points=self.lama.fit_k_elbow(k_max=self.k_max,motif_length=self.motif_length,plot_elbows=self.plot,plot_motifsets=self.plot)

    @property
    def prediction_mask_(self)->np.ndarray: 
        n_motifs=self.elbow_points.shape[0]
        mask=np.zeros((n_motifs,self.n))
        for i in range(self.elbow_points.shape[0]):
            elbow=self.elbow_points[i]
            motif_starts=self.leitmotifs[elbow]
            for j in range(motif_starts.shape[0]):
                mask[i,motif_starts[j]:motif_starts[j]+self.motif_length]=1
        return mask
    
    @property
    def prediction_dimension_(self)->list:
        dimensions=[]
        for elbow in self.elbow_points:
            dimensions.append(self.lama.leitmotifs_dims[elbow].tolist())
        return dimensions
        
        