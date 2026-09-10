import numpy as np
import itertools as it
#from sklearn.metrics import adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment


from multivariate_tsmd.utils import transform_label

####################################################################################################################
####################################################################################################################
# Sample Based
####################################################################################################################
####################################################################################################################

def b_s_precision_(R:np.ndarray,P:np.ndarray)->float: 
    """compute precision

    Args:
        R (np.ndarray): Real Label array, shape: (L,)
        P (np.ndarray): Predicted label array, shape: (L,) 

    Returns:
        float: precision score
    """
    if sum(P) != 0:
        return np.sum(R*P)/np.sum(P)
    else: 
        return 0.0

def b_s_recall_(R:np.ndarray,P:np.ndarray)->float: 
    """compute recall

    Args:
        R (np.ndarray): Real Label array, shape: (L,)
        P (np.ndarray): Predicted label array, shape: (L,) 

    Returns:
        float: recall score
    """
    if sum(R) != 0: 
        return np.sum(R*P)/np.sum(R)
    else: 
        return 0.0

def b_f1_score_(R:np.ndarray,P:np.ndarray)->float: 
    """compute f1score

    Args:
        R (np.ndarray): Real Label array, shape: (L,)
        P (np.ndarray): Predicted label array, shape: (L,) 

    Returns:
        float: f1 score
    """
    a = b_s_precision_(R,P)
    b = b_s_recall_(R,P)
    if a+b == 0:
        return 0
    else:
        return 2*a*b/(a+b)

class SampleScore:

    def __init__(self,averaging="macro") -> None:
        """Initialization

        Args:
            averaging (str, optional): macro or weigthed. Defaults to "macro".
        """
        self.averaging = averaging

    def score(self,R:np.ndarray,P:np.ndarray)->None: 
        """Compute the scoring

        Args:
            R (np.ndarray): Real Label array, shape: (N,L)
            P (np.ndarray): Predicted label array, shape: (M,L) 
        """
        N = R.shape[0]
        M = P.shape[0]

        if self.averaging == "macro": 
            self.weigths_= np.ones(N,dtype=float)/N
        elif self.averaging == "weighted": 
            weigths = np.sum(R,axis=1)
            self.weigths_ = weigths/np.sum(weigths)

        overlaps = np.zeros((N,M))
        for i,r in enumerate(R):
            for j, p in enumerate(P): 
                overlaps[i,j] = np.sum(r*p)

        self.best_cb_ = np.vstack((linear_sum_assignment(overlaps,True))).T

        #compute score intervals for best permutation
        precision = 0
        recall = 0
        f1_score = 0
        for i,j in self.best_cb_: 
            t_precision = b_s_precision_(R[i],P[j]) * self.weigths_[i]
            precision += t_precision
            t_recall = b_s_recall_(R[i],P[j]) * self.weigths_[i]
            recall += t_recall
            if t_precision + t_recall != 0: 
                f1_score += 2*t_recall*t_precision/(t_recall+t_precision)
        return precision, recall, f1_score
    
####################################################################################################################
####################################################################################################################
# Event Based
####################################################################################################################
####################################################################################################################

def event_assignement(R_lst:list,P_lst:list)->np.ndarray:
    """Assigned predicted events to real events

    Args:
        R_lst (list): Real label start end list
        P_lst (list): Predicted label start end list

    Returns:
        np.ndarray : pairing
    """ 
    #compute TP Matrix
    n_r = len(R_lst)
    n_p = len(P_lst)
    arr = np.zeros((n_r,n_p))
    for i,(rs,re) in enumerate(R_lst): 
        for j, (ps,pe) in enumerate(P_lst): 

            s = max(rs,ps)
            e = min(re,pe)
            if e>=s: 
                arr[i,j] = e-s+1

    pairing = np.vstack((linear_sum_assignment(arr,True))).T
    return pairing

def b_e_precision_(R_lst:list,P_lst:list,threshold=0.9)->float: 
    """Event based precision

    Args:
        R_lst (list): Real label start end list
        P_lst (list): Predicted label start end list
        threshold (float or np.ndarray, optional): threshold to evaluate. Defaults to 0.9.

    Returns:
        float or np.ndarray : precision score
    """
    if isinstance(threshold,float):
        if len(R_lst)*len(P_lst) !=0:
            pairing = event_assignement(R_lst,P_lst)
            Rp = R_lst[pairing[:,0]]
            Pp = P_lst[pairing[:,1]]
            score = 0 
            for (rs,re),(ps,pe) in zip(Rp,Pp): 
                s = max(rs,ps)
                e = min(re,pe)
                if e-s>=threshold*(pe-ps): 
                    score+=1
            return score/len(P_lst) 
        else: 
            return 0.0
    elif isinstance(threshold,np.ndarray): 
        scores = np.zeros_like(threshold)
        if len(R_lst)*len(P_lst) !=0:
            pairing = event_assignement(R_lst,P_lst)
            Rp = R_lst[pairing[:,0]]
            Pp = P_lst[pairing[:,1]]
            for (rs,re),(ps,pe) in zip(Rp,Pp): 
                s = max(rs,ps)
                e = min(re,pe)
                idxs = np.where(threshold<= (e-s)/(pe-ps)) 
                scores[idxs] +=1
            return scores/len(P_lst) 
        else: 
            return scores
    else: 
        raise ValueError("Threshold must be a float or an numpy.ndarray")

def b_e_recall_(R_lst:list,P_lst:list,threshold=0.9)->float: 
    """Event based precision

    Args:
        R_lst (list): Real label start end list
        P_lst (list): Predicted label start end list
        threshold (float or np.ndarray, optional): threshold to evaluate. Defaults to 0.9.

    Returns:
        float or np.ndarray: recall score
    """
    if isinstance(threshold,float):
        if len(R_lst)*len(P_lst) != 0:
            pairing = event_assignement(R_lst,P_lst)
            Rp = R_lst[pairing[:,0]]
            Pp = P_lst[pairing[:,1]]
            score = 0 
            for (rs,re),(ps,pe) in zip(Rp,Pp): 
                s = max(rs,ps)
                e = min(re,pe)
                if e-s>=threshold*(re-rs): 
                    score+=1
            return score/len(R_lst)
        else: 
            return 0.
    elif isinstance(threshold,np.ndarray): 
        scores = np.zeros_like(threshold)
        if len(R_lst)*len(P_lst) !=0:
            pairing = event_assignement(R_lst,P_lst)
            Rp = R_lst[pairing[:,0]]
            Pp = P_lst[pairing[:,1]]
            for (rs,re),(ps,pe) in zip(Rp,Pp): 
                s = max(rs,ps)
                e = min(re,pe)
                idxs = np.where(threshold<= (e-s)/(re-rs)) 
                scores[idxs] +=1
            return scores/len(R_lst) 
        else: 
            return scores
    else: 
        raise ValueError("Threshold must be a float or an numpy.ndarray")

def dim_precision_(r_dims: list, p_dims: list):

    r_set = set(r_dims)
    p_set = set(p_dims)

    tp = len(r_set & p_set)         # correctement trouvées
    fp = len(p_set - r_set)         # fausses dimensions détectées

    if tp + fp == 0:
        return 0.0
    
    return tp / (tp + fp)

def dim_recall_(r_dims: list, p_dims: list):
    """
    Recall dimensionnel = TP / (TP + FN)
    """
    r_set = set(r_dims)
    p_set = set(p_dims)

    tp = len(r_set & p_set)         # correctement trouvées
    fn = len(r_set - p_set)         # dimensions manquées

    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def dim_f1_(r_dims, p_dims):
    p = dim_precision_(r_dims, p_dims)
    r = dim_recall_(r_dims, p_dims)

    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def b_e_f1_score_(R_lst:list,L_lst:list,threshold=0.9)->float: 
    """Event based precision

    Args:
        R_lst (list): Real label start end list
        P_lst (list): Predicted label start end list
        threshold (float or np.ndarray, optional): threshold to evaluate. Defaults to 0.9.

    Returns:
        float or np.ndarray: precision score
    """
    if isinstance(threshold,float): 
        a = b_e_precision_(R_lst,L_lst,threshold)
        b = b_e_recall_(R_lst,L_lst,threshold)
        if a+b == 0:
            return 0
        else:
            return 2*a*b/(a+b)
    elif isinstance(threshold,np.ndarray): 
        a = b_e_precision_(R_lst,L_lst,threshold)
        b = b_e_recall_(R_lst,L_lst,threshold)
        scores = np.zeros_like(threshold)
        idxs = np.where(a+b != 0)
        scores[idxs] = 2*a[idxs]*b[idxs]/(a[idxs]+b[idxs])
        return scores
    else: 
        raise ValueError("Threshold must be a float or an numpy.ndarray")
    
class EventScore(object):

    def __init__(self,averaging = "macro") -> None:
        """Intialization

        Args:
            nbins (int, optional): number of bins to compute the AUC. Defaults to 100.
            averaging (str,optional): macro or weighted. Defaults to "macro".
        """
        self.averaging = averaging

    def _find_best_permutation(self,R:np.ndarray,P:np.ndarray): 

        N = R.shape[0]
        M = P.shape[0]
        overlaps = np.zeros((N,M))
        for i,r in enumerate(R):
            for j, p in enumerate(P): 
                overlaps[i,j] = np.sum(r*p)

        self.best_cb_ = np.vstack((linear_sum_assignment(overlaps,True))).T


    def score(self,R:np.ndarray,P:np.ndarray,threshold=0.1)->tuple:
         
        R_lst = transform_label(R)
        P_lst = transform_label(P)
        N = R.shape[0]
        if self.averaging == "macro": 
            self.weigths_= np.ones(N,dtype=float)/N
        elif self.averaging == "weighted": 
            weigths = np.array([len(l) for l in R_lst],dtype=float)
            self.weigths_ = weigths/np.sum(weigths)

        self._find_best_permutation(R,P)

        # compute score:
        if isinstance(threshold,float): 
            precision = 0
            recall = 0 
            f1_score = 0
            for i,j in self.best_cb_:
                r = R_lst[i]
                p = P_lst[j]
                t_precision = b_e_precision_(r,p,threshold) * self.weigths_[i]
                t_recall = b_e_recall_(r,p,threshold) * self.weigths_[i]
                precision += t_precision 
                recall += t_recall 
                if t_precision + t_recall != 0: 
                    f1_score += 2*t_precision*t_recall / (t_precision + t_recall)
        elif isinstance(threshold,np.ndarray): 
            precision = np.zeros_like(threshold)
            recall = np.zeros_like(threshold)
            f1_score = np.zeros_like(threshold)
            for i,j in self.best_cb_:
                r = R_lst[i]
                p = P_lst[j]
                t_precision = b_e_precision_(r,p,threshold) * self.weigths_[i]
                t_recall = b_e_recall_(r,p,threshold) * self.weigths_[i]
                precision += t_precision 
                recall += t_recall 
                idxs = np.where(t_precision+t_recall != 0)
                f1_score[idxs] += 2*t_precision[idxs]*t_recall[idxs]/(t_precision[idxs]+t_recall[idxs])
        else: 
            raise ValueError("Threshold must be a float or an numpy.ndarray")
                 
        return precision, recall, f1_score
    
class EventScoreSubDims(object):

    def __init__(self,averaging = "macro") -> None:
        """Intialization

        Args:
            nbins (int, optional): number of bins to compute the AUC. Defaults to 100.
            averaging (str,optional): macro or weighted. Defaults to "macro".
        """
        self.averaging = averaging

    def _find_best_permutation(self,R:np.ndarray,P:np.ndarray,R_dims:list,P_dims:list): 

        N = R.shape[0]
        M = P.shape[0]
        overlaps = np.zeros((N,M))
        for i,r in enumerate(R):
            r_dims=R_dims[i]
            for j, p in enumerate(P): 
                p_dims=P_dims[j]
                time_overlap = np.sum(r*p)
                inter_d = len(set(r_dims) & set(p_dims))
                union_d = len(set(r_dims) | set(p_dims))
                if union_d > 0:
                    dim_overlap = inter_d / union_d 
                else:
                    dim_overlap=0
                overlaps[i,j]=time_overlap*dim_overlap

        self.best_cb_ = np.vstack((linear_sum_assignment(overlaps,True))).T

    def score(self,R:np.ndarray,P:np.ndarray,R_dims, P_dims,threshold=0.1)->tuple:
        R_lst = transform_label(R)
        P_lst = transform_label(P)
        N = R.shape[0]
        if self.averaging == "macro": 
            self.weigths_= np.ones(N,dtype=float)/N
        elif self.averaging == "weighted": 
            weigths = np.array([len(l) for l in R_lst],dtype=float)
            self.weigths_ = weigths/np.sum(weigths)

        self._find_best_permutation(R,P,R_dims,P_dims)


        # compute score:
        if isinstance(threshold,float): 
            precision = 0
            recall = 0 
            f1_score = 0
            dim_precision = 0 
            dim_recall = 0
            dim_f1 = 0
            for i,j in self.best_cb_:
                r = R_lst[i]
                p = P_lst[j]
                r_dims=R_dims[i]
                p_dims=P_dims[j]

                t_precision = b_e_precision_(r,p,threshold) * self.weigths_[i]
                t_recall = b_e_recall_(r,p,threshold) * self.weigths_[i]
                d_precision = dim_precision_(r_dims, p_dims) * self.weigths_[i]
                d_recall = dim_recall_(r_dims, p_dims) * self.weigths_[i]

                precision += t_precision 
                recall += t_recall 
                dim_precision += d_precision
                dim_recall += d_recall

                if t_precision + t_recall != 0: 
                    f1_score += 2*t_precision*t_recall / (t_precision + t_recall)
                if d_precision + d_recall != 0:
                    dim_f1 += 2*d_precision*d_recall / (d_precision + d_recall)

        elif isinstance(threshold,np.ndarray): 
            precision = np.zeros_like(threshold)
            recall = np.zeros_like(threshold)
            f1_score = np.zeros_like(threshold)
            dim_precision = 0 
            dim_recall = 0
            dim_f1 = 0 
            for i,j in self.best_cb_:
                r = R_lst[i]
                p = P_lst[j]
                r_dims=R_dims[i]
                p_dims=P_dims[j]
                
                t_precision = b_e_precision_(r,p,threshold) * self.weigths_[i]
                t_recall = b_e_recall_(r,p,threshold) * self.weigths_[i]
                d_precision = dim_precision_(r_dims, p_dims) * self.weigths_[i]
                d_recall = dim_recall_(r_dims, p_dims) * self.weigths_[i]
                d_f1 = dim_f1_(r_dims, p_dims) * self.weigths_[i]

                precision += t_precision 
                recall += t_recall 

                dim_precision += d_precision
                dim_recall += d_recall
                dim_f1 += d_f1
                idxs = np.where(t_precision+t_recall != 0)
                f1_score[idxs] += 2*t_precision[idxs]*t_recall[idxs]/(t_precision[idxs]+t_recall[idxs])
            
            dim_precision = dim_precision*np.ones_like(threshold)
            dim_recall = dim_recall*np.ones_like(threshold)
            dim_f1 = dim_f1*np.ones_like(threshold)
        else: 
            raise ValueError("Threshold must be a float or an numpy.ndarray")

        return precision, recall, f1_score, dim_precision, dim_recall, dim_f1


####################################################################################################################
####################################################################################################################
# Clustering Based
####################################################################################################################
####################################################################################################################


class AdjustedMutualInfoScore:

    def __init__(self) -> None:
        pass

    def score(self,R:np.ndarray,P:np.ndarray)->None: 
        """Compute scoring

        Args:
            label (np.ndarray): label, shape (N,L)
            predictions (np.ndarray): prediction, shape (N,L)

        Returns:
            np.ndarray: score
        """ 
        n_R = R.shape[0]
        R_label = np.sum(R * np.arange(1,n_R+1).reshape(-1,1),axis = 0)
        n_P = P.shape[0]
        P_label = np.sum(P * np.arange(1,n_P+1).reshape(-1,1),axis = 0)
        return adjusted_mutual_info_score(R_label,P_label)
