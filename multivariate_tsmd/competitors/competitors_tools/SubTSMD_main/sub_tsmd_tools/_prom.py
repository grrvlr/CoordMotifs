""" https://github.com/ML-KULeuven/tsmd-evaluation/blob/main/tsmd_evaluation/prom.py """

import numpy as np
import scipy.optimize


def overlap_rate(s1, e1, s2, e2):
    """
    Calculate the overlap rate between two segments.

    Parameters:
    s1 (int): Start of the first interval.
    e1 (int): End of the first interval.
    s2 (int): Start of the second interval.
    e2 (int): End of the second interval.

    Returns:
    float: The overlap rate between the two intervals. It is calculated as the overlap divided by their union.
    """
    return max(0, (min(e1, e2) - max(s1, s2)) / (max(e1, e2) - min(s1, s2)))


def overlap_rate_multivariate_segments(segment1, segment2):
    """
    Compute the minimum overlap rate between motif locations across all
    attributes.

    :param segment1: array of shape (2, n_attributes)
    :param segment2: array of shape (2, n_attributes)

    :return: The minimum overlap rate across all dimensions.
    """
    return np.min(
        [
            overlap_rate(s1, e1, s2, e2)
            for (s1, e1), (s2, e2) in zip(segment1.T, segment2.T)
        ]
    )


def subspaces_overlap(mask1, mask2, threshold):
    """
    Check whether the given two masks overlap sufficiently based on the Jaccard similarity.

    :param mask1: Mask of the first motif
    :param mask2: Mask of the second motif
    :param threshold: Threshold on the jaccard similarity to have overlap

    :return: Whether the Jaccard similarity of the two masks exceeds the given threshold
    """
    return ((mask1 & mask2).sum() / (mask1 | mask2).sum()) >= threshold


def matching_matrix(gt, discovered_sets, threshold_subspace=1.0, threshold_ovr=0.5):
    """
    Create a match matrix between ground truth and discovered sets based on overlap rate.

    Parameters:
    gt (list or dict): Ground truth intervals.
    discovered_sets (list): Discovered intervals.
    threshold_ovr (float): Overlap threshold_ovr to consider a match.

    Returns:
    tuple: Match matrix, row names, and column names.
    """
    assert threshold_ovr >= 0.5
    assert gt

    g, d = len(gt), len(discovered_sets)
    column_names = np.arange(1, d + 1)
    if type(gt) is list:
        row_names = np.arange(1, g + 1)
        gt_sets = gt
    elif type(gt) is dict:
        row_names = np.array(list(gt.keys()))
        gt_sets = list(gt.values())

    # Edge case: no discovered motif sets
    if not discovered_sets:
        # In this case only return the unmatched GT motifs column
        match_matrix = np.zeros((g + 1, 1), dtype=int)
        match_matrix[:-1, 0] = [len(gt_set) for gt_set in gt_sets]
        match_matrix[-1, 0] = 0
        return match_matrix, row_names, np.array([])

    # Contingency  table M
    # ADJUSTED: iteration variables are changed to correspond to subspace motifs data structure
    # ADJUSTED: a check to see if the masks of the two motif sets align
    # ADJUSTED: the wrapper overlap rate function is used, which computes the average overlap across the dimensions.
    mm = np.zeros((g, d), dtype=int)
    for i, (gt_mask, gt_locations) in enumerate(gt_sets):
        for gt_segment in gt_locations:

            # Find the best match in any motif set, greater than threshold
            best = None
            best_ovr = 0.0
            for j, (discovered_mask, discovered_locations) in enumerate(
                discovered_sets
            ):
                # Skip if the mask does not align
                if not subspaces_overlap(gt_mask, discovered_mask, threshold_subspace):
                    continue

                for discovered_segment in discovered_locations:
                    ovr = overlap_rate_multivariate_segments(
                        gt_segment, discovered_segment
                    )
                    if ovr > threshold_ovr and ovr > best_ovr:
                        best = j
                        best_ovr = ovr

            if best is not None:
                mm[i, best] += 1

    # Optimally match GT and discovered motif sets
    r, c = scipy.optimize.linear_sum_assignment(mm, maximize=True)

    r_perm = np.hstack((r, np.setdiff1d(range(g), r)))
    c_perm = np.hstack((c, np.setdiff1d(range(d), c)))

    # Compute number of unmatched GT motifs per set
    # ADJUSTED: iteration variables are changed to correspond to subspace motifs data structure
    md = np.array([gt_set.shape[0] for (_, gt_set) in gt_sets])
    md = md - np.sum(mm, axis=1)

    # compute number of unmatched discovered motifs per set
    # ADJUSTED: iteration variables are changed to correspond to subspace motifs data structure
    fd = np.array([discovered_set.shape[0] for (_, discovered_set) in discovered_sets])
    fd = fd - np.sum(mm, axis=0)

    # Permute M according to the best permutation
    mm = mm[:, c_perm]
    mm = mm[r_perm, :]

    # Permute fd and md as well
    fd = fd[c_perm]
    md = md[r_perm]

    # Construct the matching matrix
    md = np.expand_dims(md, axis=0).T
    M = np.block([[mm, md], [fd, np.nan]])
    # Check ties
    if tie(M):
        M, r_perm, c_perm = break_ties(M, r_perm, c_perm)

    return M, row_names[r_perm], column_names[c_perm]


def tie(M):
    """
    Check if there is a tie (an alternative permutation with the same number of total true positives) in the match matrix.

    Parameters:
    M (numpy.ndarray): Match matrix.

    Returns:
    bool: True if there is a tie, False otherwise.
    """
    (g, d) = M.shape[0] - 1, M.shape[1] - 1
    if g >= d:
        return False

    for i in range(g):
        tp = M[i, i]
        if np.sum(M[i, :d] == tp) > 1:
            return True
    return False


# When there are multiple optimal permutations, tie break based on the number of unmatched discovered motifs
# Note: We only do so when d >= g. # Ties when g > d are also possible, but this will only affect the macro averaged recall. We do not consider them (for now).
# In that case, ties need to be broken based on cardinality of the GT motif sets (smaller ones are to be matched). Consider for example M = [[2 0], [2 2], [0 0]]
def break_ties(M, r, c):
    """
    Break ties in the match matrix based on the number of unmatched discovered motifs.

    Parameters:
    M (numpy.ndarray): Match matrix.
    r (numpy.ndarray): Row indices.
    c (numpy.ndarray): Column indices.

    Returns:
    tuple: Updated match matrix, row names, and column names.
    """
    g, d, _ = get_g_d_m(M)
    mm = np.zeros((g, d))

    for i in range(g):
        tp = M[i, i]
        fp = np.sum(M[:, :d], axis=0, dtype=float) - M[i, :d]
        fp[~(M[i, :d] == tp)] = np.inf
        mm[i, :] = fp

    r_, c_ = scipy.optimize.linear_sum_assignment(mm, maximize=False)

    r_ = np.concatenate((r_, [g]))
    c_ = np.concatenate((c_, np.setdiff1d(range(d), c_), [d]))

    M = M[r_, :]
    M = M[:, c_]
    return M, r[r_[:g]], c[c_[:d]]


def get_g_d_m(M):
    """
    Get the dimensions of the matching matrix and the number of matched motif sets.

    Parameters:
    M (numpy.ndarray): Matching matrix.

    Returns:
    tuple: Number of ground truth sets, number of discovered sets, and the number of matched pairs of sets.
    """
    g, d = M.shape[0] - 1, M.shape[1] - 1
    return g, d, min(g, d)


## MICRO-AVERAGED METRICS
def micro_averaged_recall(M):
    g, d, m = get_g_d_m(M)
    if d == 0:
        return 0.0
    diag = np.diag(M[:m, :m])
    return np.sum(diag) / np.sum(M[:g, :])


def micro_averaged_precision(M, penalize_off_target=False):
    _, d, m = get_g_d_m(M)

    if d == 0:
        return np.nan

    diag = np.diag(M[:m, :m])
    if penalize_off_target:
        return np.sum(diag) / np.sum(M[:, :d])
    else:
        return np.sum(diag) / np.sum(M[:, :m])


def micro_averaged_f1(M, penalize_off_target=False):
    p = micro_averaged_precision(M, penalize_off_target=penalize_off_target)
    r = micro_averaged_recall(M)
    if (p == 0.0 and r == 0.0) or np.isnan(p):
        return 0.0
    return 2 * r * p / (p + r)


## MACRO-AVERAGED METRICS
def macro_averaged_precision(M, penalize_off_target=False):
    _, d, m = get_g_d_m(M)

    if d == 0:
        return np.nan

    tps = np.diag(M[:m, :m])
    ps = np.divide(tps, np.sum(M[:, :m], axis=0))
    p = np.sum(ps) / d if penalize_off_target else np.sum(ps) / m
    return p


def macro_averaged_recall(M):
    g, _, m = get_g_d_m(M)

    tps = np.diag(M[:m, :m])
    rs = np.divide(tps, np.sum(M[:m, :], axis=1))
    r = np.sum(rs) / g
    return r


def macro_averaged_f1(M, penalize_off_target=False):
    g, d, m = get_g_d_m(M)
    tps = np.diag(M[:m, :m])

    if np.sum(tps) == 0:
        return 0.0

    # Calculate recall and precision for matched motif sets
    rs = np.divide(tps, np.sum(M[:m, :], axis=1))
    ps = np.divide(tps, np.sum(M[:, :m], axis=0))

    fs = 2 * np.divide(np.multiply(rs, ps), (rs + ps))
    fs[(rs == 0.0) & (ps == 0.0)] = 0

    # Handle unmatched motif sets
    if d >= g:
        f = np.sum(fs) / (d if penalize_off_target else m)
    else:
        f = np.sum(fs) / g

    return f
