import itertools
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Core scoring / graph utilities
# ---------------------------------------------------------------------------

def asymmetric_overlap_score(cc_i, cc_j):
    Pi = set(cc_i[1])
    Pj = set(cc_j[1])
    if len(Pi) == 0:
        return 0.0
    return len(Pi & Pj) / len(Pi)


def build_directed_cc_graph(candidate_sets, min_weight=0.8):
    """Build directed weighted graph between connected candidates.

    Args:
        candidate_sets: dict {seed: (dim_list, pos_list)}
        min_weight: minimal directed overlap score to keep an edge

    Returns:
        networkx.DiGraph
    """
    G = nx.DiGraph()
    seeds = list(candidate_sets.keys())
    G.add_nodes_from(seeds)

    for i, j in itertools.permutations(seeds, 2):
        w = asymmetric_overlap_score(candidate_sets[i], candidate_sets[j])
        if w >= min_weight:
            G.add_edge(i, j, weight=w)
    return G


def gather_scc(candidate_sets, G):
    """Merge nodes belonging to the same SCC by intersecting their positions.
 
    Singleton SCCs keep their original key; merged SCCs get a tuple key.
 
    Args:
        candidate_sets: dict {seed: (dim_list, pos_list)}
        G: directed graph of connected candidates
        n_patterns: number of top patterns to use for coverage computation.
    """
    # Separate singleton and non-singleton SCCs
    singleton_sccs = []
    non_singleton_sccs = []
    for scc in nx.strongly_connected_components(G):
        if len(scc) == 1:
            singleton_sccs.append(scc)
        else:
            non_singleton_sccs.append(scc)
 
    new_candidate_sets = {}
 
    # Singletons are always kept as-is
    for scc in singleton_sccs:
        node = next(iter(scc))
        new_candidate_sets[node] = candidate_sets[node]

        # Original behaviour: merge every non-singleton SCC unconditionally
    for scc in non_singleton_sccs:
        positions = None
        dimensions = set()
        for node in scc:
            node_dimensions, node_positions = candidate_sets[node]
            positions = set(node_positions) if positions is None else positions & set(node_positions)
            dimensions.update(node_dimensions)
        new_candidate_sets[tuple(scc)] = (dimensions, positions)
    return new_candidate_sets


def compute_candidate_coverage(candidate_sets):
    """Return a dict {cc_id: coverage_score} sorted by descending coverage."""
    coverage_scores = {
        cc_id: len(dimensions) * len(positions)
        for cc_id, (dimensions, positions) in candidate_sets.items()
    }
    return dict(sorted(coverage_scores.items(), key=lambda x: x[1], reverse=True))

def optimal_n_patterns(coverage_scores):
    values = np.array(list(coverage_scores.values()), dtype=float)
    if len(values) <= 1:
        return len(values)
    
    values_with_zero = np.append(values, 0)
    return int(np.argmax(abs(np.diff(values_with_zero)))) + 1


def compute_coverage(coverage_scores, n_patterns):
    if n_patterns is None:
        n_patterns = optimal_n_patterns(coverage_scores)
    return sum(v for _, v in list(coverage_scores.items())[:n_patterns])


def get_sink_and_ancestors_sccs(G, coverage_scores):
    """Return sinks (with at least one ancestor) and their ancestors,
    both sorted by descending coverage score."""
    sink_ancestor_pairs = []
    for sink_id in (n for n in G.nodes if G.out_degree(n) == 0):
        anc_ids = nx.ancestors(G, sink_id)
        if not anc_ids:
            continue
        anc_ids_sorted = sorted(anc_ids, key=lambda n: coverage_scores.get(n, 0), reverse=True)
        sink_ancestor_pairs.append((sink_id, anc_ids_sorted))

    sink_ancestor_pairs.sort(key=lambda x: coverage_scores.get(x[0], 0), reverse=True)

    if not sink_ancestor_pairs:
        return [], []
    sinks, ancestors = zip(*sink_ancestor_pairs)
    return list(sinks), list(ancestors)


# ---------------------------------------------------------------------------
# Split utilities
# ---------------------------------------------------------------------------

def nodes_intersection(node_1, node_2, candidates):
    dims_1, positions_1 = candidates[node_1]
    dims_2, positions_2 = candidates[node_2]

    positions_1, positions_2 = set(positions_1), set(positions_2)
    intersection = positions_1 & positions_2
    total_dims = set(dims_1) | set(dims_2)

    return (
        (total_dims, sorted(intersection)),
        (dims_1, sorted(positions_1 - intersection)),
        (dims_2, sorted(positions_2 - intersection)),
    )


def apply_split(node_1, node_2, candidate_sets):
    """Split node_1 (sink) using node_2 (ancestor); return updated candidates dict."""
    intersection, residual_1, residual_2 = nodes_intersection(node_1, node_2, candidate_sets)

    new_candidate_sets = dict(candidate_sets)
    del new_candidate_sets[node_1]
    del new_candidate_sets[node_2]

    if intersection[1]:
        new_candidate_sets[(node_1, node_2, "inter")] = intersection
    if residual_1[1]:
        new_candidate_sets[(node_1, "res1")] = residual_1
    if residual_2[1]:
        new_candidate_sets[(node_2, "res2")] = residual_2

    return new_candidate_sets


# ---------------------------------------------------------------------------
# InterdimConnections
# ---------------------------------------------------------------------------

class InterdimConnections:

    def __init__(self, min_weight=0.8, n_patterns=None, save_history=False):
        self.min_weight = min_weight
        self.n_patterns = n_patterns
        self.save_history = save_history

    # ------------------------------------------------------------------
    # Graph / SCC pipeline
    # ------------------------------------------------------------------

    def compute_graph_and_scc(self, candidate_sets):
        initial_G = build_directed_cc_graph(candidate_sets, self.min_weight)
        current_candidates = gather_scc(
            candidate_sets, initial_G,
        )
        coverage_scores = compute_candidate_coverage(current_candidates)
        coverage = compute_coverage(coverage_scores, self.n_patterns)

        final_G = build_directed_cc_graph(current_candidates, self.min_weight)
        sinks_list, ancestors_list = get_sink_and_ancestors_sccs(final_G, coverage_scores)

        return initial_G, final_G, current_candidates, coverage_scores, coverage, sinks_list, ancestors_list

    def evaluate_split_trial(self, node_1, node_2, current_candidates, base_coverage):
        split_candidates = apply_split(node_1, node_2, current_candidates)

        initial_G, final_G, new_candidates, coverage_scores, coverage, sinks_list, ancestors_list = \
            self.compute_graph_and_scc(split_candidates)

        return {
            "sink": node_1,
            "ancestor": node_2,
            "split_candidates": split_candidates,   # avant MergeSCC
            "candidates": new_candidates,           # après MergeSCC
            "initial_G": initial_G,
            "G": final_G,
            "coverage_scores": coverage_scores,
            "coverage": coverage,
            "gain": coverage - base_coverage,
            "sinks": sinks_list,
            "ancestors": ancestors_list,
        }
    
    def get_best_split(self, base_coverage, sinks_list, ancestors_list, current_candidates):
        best_gain = 0
        best_trial = None
        split_trials = []

        for i, node_1 in enumerate(sinks_list):
            for node_2 in ancestors_list[i]:
                trial = self.evaluate_split_trial(
                    node_1,
                    node_2,
                    current_candidates,
                    base_coverage,
                )

                split_trials.append(trial)

                if trial["gain"] > best_gain:
                    best_gain = trial["gain"]
                    best_trial = trial

        if best_trial is None:
            return 0, None, None, None, None, None, None, split_trials

        return (
            best_trial["gain"],
            best_trial["initial_G"],
            best_trial["G"],
            best_trial["candidates"],
            best_trial["coverage_scores"],
            best_trial["sinks"],
            best_trial["ancestors"],
            split_trials,
        )

    # ------------------------------------------------------------------
    # Iterative split loop
    # ------------------------------------------------------------------

    def iterative_split(self):
        self.n_iterations = 0
        if self.save_history:
            self.history = []

        while True:
            if self.save_history:
                n_patterns = self.n_patterns if self.n_patterns is not None else optimal_n_patterns(self.coverage_scores)
                self.history.append({
                    "candidates": self.current_candidates,
                    "initial_G": self.current_initial_G,
                    "G": self.current_G,
                    "coverage": self.coverage,
                    "coverage_scores": self.coverage_scores,
                    "n_patterns": n_patterns,
                })

            best_gain, best_initial_G, best_G, best_candidates, best_coverage_scores, best_sinks, best_ancestors, split_trials = \
                self.get_best_split(self.coverage, self.sinks_list, self.ancestors_list, self.current_candidates)
            if self.save_history:
                self.history[-1]["split_trials"] = split_trials
            if best_gain <= 0:
                break

            self.coverage += best_gain
            self.current_candidates = best_candidates
            self.coverage_scores = best_coverage_scores
            self.current_initial_G = best_initial_G
            self.current_G = best_G
            self.sinks_list, self.ancestors_list = best_sinks, best_ancestors
            self.n_iterations += 1

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, univariate_candidates):
        self.univariate_candidates = univariate_candidates
        self.current_initial_G, self.current_G, self.current_candidates, self.coverage_scores, self.coverage, \
            self.sinks_list, self.ancestors_list = self.compute_graph_and_scc(self.univariate_candidates)
        self.iterative_split()
        if self.n_patterns is None:
            self.n_patterns = optimal_n_patterns(self.coverage_scores)

        effective_n = min(self.n_patterns, len(self.coverage_scores))
        top_keys = list(self.coverage_scores.keys())[:effective_n]
        self.final_candidates = {i: self.current_candidates[k] for i, k in enumerate(top_keys)}