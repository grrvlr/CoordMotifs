import itertools
import networkx as nx
import numpy as np
from kneed import KneeLocator

from collections import defaultdict

def rename_initial_components(components):
    dim_counters = defaultdict(int)
    new_components = {}
    next_id = 0

    for _, (dims, pos) in components.items():
        dim = dims[0]
        i = dim_counters[dim]
        dim_counters[dim] += 1

        new_components[next_id] = {
            "dims": set(dims),
            "pos": sorted(pos),
            "label": f"({dim},{i})"
        }
        next_id += 1

    return new_components

def relabel_components(components):
    """
    Re-assign labels of the form ({dims}, i)
    where i is stable within each dims group,
    ordered by decreasing component size.
    """
    from collections import defaultdict

    dim_groups = defaultdict(list)

    # regrouper par ensemble de dims
    for cid, comp in components.items():
        key = tuple(sorted(comp["dims"]))
        dim_groups[key].append((cid, comp))

    new_components = {}

    for dims, comps in dim_groups.items():
        comps_sorted = sorted(
            comps,
            key=lambda x: len(x[1]["pos"]),
            reverse=True
        )

        for i, (cid, comp) in enumerate(comps_sorted):
            new_components[cid] = {
                "dims": comp["dims"],
                "pos": comp["pos"],
                "label": f"({set(dims)}, {i})"
            }

    return new_components

# ---------------------------------------------------------------------------
# Core scoring / graph utilities
# ---------------------------------------------------------------------------

def asymetric_overlap_score(cc_i, cc_j):
    Pi = set(cc_i["pos"])
    Pj = set(cc_j["pos"])
    if len(Pi) == 0:
        return 0.0
    return len(Pi & Pj) / len(Pi)


def build_directed_cc_graph(components, min_weight=0.8):
    """Build directed weighted graph between connected components.

    Args:
        components: dict {seed: (dim_list, pos_list)}
        min_weight: minimal directed overlap score to keep an edge

    Returns:
        networkx.DiGraph
    """
    G = nx.DiGraph()
    seeds = list(components.keys())
    G.add_nodes_from(seeds)

    for i, j in itertools.permutations(seeds, 2):
        w = asymetric_overlap_score(components[i], components[j])
        if w >= min_weight:
            G.add_edge(i, j, weight=w)
    return G

def scc_residuals(scc, components, intersection_positions):
    """Return per-node residual components for a gathered SCC.
 
    For each node whose positions are not fully covered by the intersection,
    a residual component is created holding only the positions outside the
    intersection. The node's own dimensions are preserved on its residual.
 
    Args:
        scc: iterable of node keys belonging to the SCC
        components: dict {seed: (dim_list, pos_list)}
        intersection_positions: set of positions kept in the merged component
 
    Returns:
        dict of {(node, "res"): (dims, sorted_residual_positions)} for nodes
        that have at least one position outside the intersection.
    """
    residuals = {}
    for node in scc:
        node_dims, node_positions = components[node]
        leftover = set(node_positions) - intersection_positions
        if leftover:
            residuals[(node, "res")] = (node_dims, sorted(leftover))
    return residuals


def gather_scc(components, G, n_patterns=None, arbitrate=False, keep_residuals=False):
    """Merge nodes belonging to the same SCC by intersecting their positions.
 
    Singleton SCCs keep their original key; merged SCCs get a tuple key.
 
    Args:
        components: dict {seed: (dim_list, pos_list)}
        G: directed graph of connected components
        n_patterns: number of top patterns to use for coverage computation
                    when arbitrate=True. If None, optimal_n_patterns is used.
        arbitrate: if True, a non-singleton SCC is only merged when the
                   resulting coverage >= current coverage. SCCs are evaluated
                   in descending order of their max-node coverage score so
                   that the most impactful gathers are decided first, reducing
                   order-dependency. If False (default), all SCCs are always
                   merged unconditionally (original behaviour).
        keep_residuals: if True, positions that fall outside the intersection
                        of a gathered SCC are kept as separate per-node residual
                        components (keyed as (node, "res")) instead of being
                        discarded. Has no effect on refused gathers (nodes are
                        kept intact) or singletons. Independent of arbitrate.
    """
    # Separate singleton and non-singleton SCCs
    singleton_sccs = []
    non_singleton_sccs = []
    for scc in nx.strongly_connected_components(G):
        if len(scc) == 1:
            singleton_sccs.append(scc)
        else:
            non_singleton_sccs.append(scc)
 
    new_components = {}
 
    # Singletons are always kept as-is
    for scc in singleton_sccs:
        node = next(iter(scc))
        new_components[node] = components[node]
 
    if not arbitrate:
        # Original behaviour: merge every non-singleton SCC unconditionally
        for scc in non_singleton_sccs:
            positions = None
            dimensions = set()
            labels = []

            for node in scc:
                comp = components[node]
                node_dims, node_positions = comp["dims"], comp["pos"]

                positions = set(node_positions) if positions is None else positions & set(node_positions)
                dimensions.update(node_dims)
                labels.append(comp["label"])

            new_id = max(new_components.keys(), default=-1) + 1

            new_components[new_id] = {
                "dims": dimensions,
                "pos": sorted(positions),
                "label": f"M{new_id}"
            }
            if keep_residuals:
                new_components.update(scc_residuals(scc, components, positions))
        return new_components
 
    # Arbitrated path ---------------------------------------------------
    # Sort non-singleton SCCs by descending max coverage score among their
    # nodes, so the most impactful gathers are arbitrated first.
    base_coverage_scores = compute_ccp_coverage(components)
    base_coverage = compute_coverage(base_coverage_scores, n_patterns)
 
    def scc_priority(scc):
        return max(base_coverage_scores.get(node, 0) for node in scc)
 
    non_singleton_sccs.sort(key=scc_priority, reverse=True)
 
    for scc in non_singleton_sccs:
        # Build candidate merged component
        positions = None
        dimensions = set()
        for node in scc:
            node_dimensions, node_positions = components[node]
            positions = set(node_positions) if positions is None else positions & set(node_positions)
            dimensions.update(node_dimensions)
 
        merged_key = tuple(scc)
        candidate_components = {k: v for k, v in components.items() if k not in scc}
        candidate_components[merged_key] = (dimensions, positions)
        candidate_coverage_scores = compute_ccp_coverage(candidate_components)
        candidate_coverage = compute_coverage(candidate_coverage_scores, n_patterns)
 
        if candidate_coverage >= base_coverage:
            # Gather accepted; update running state for subsequent SCCs
            new_components[merged_key] = (dimensions, positions)
            if keep_residuals:
                new_components.update(scc_residuals(scc, components, positions))
            components = candidate_components
            base_coverage = candidate_coverage
        else:
            # Gather refused: keep nodes separated
            for node in scc:
                new_components[node] = components[node]
 
    return relabel_components(new_components)


def compute_ccp_coverage(connected_components):
    """Return a dict {cc_id: coverage_score} sorted by descending coverage."""
    coverage_scores = {
    cc_id: len(comp["dims"]) * len(comp["pos"])
    for cc_id, comp in connected_components.items()
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

def nodes_intersection(node_1, node_2, components):
    c1 = components[node_1]
    c2 = components[node_2]

    dims_1, positions_1 = c1["dims"], set(c1["pos"])
    dims_2, positions_2 = c2["dims"], set(c2["pos"])

    intersection = positions_1 & positions_2
    total_dims = dims_1 | dims_2

    return (
        (total_dims, intersection),
        (dims_1, positions_1 - intersection),
        (dims_2, positions_2 - intersection),
    )


'''def apply_split(node_1, node_2, components):
    intersection, residual_1, residual_2 = nodes_intersection(node_1, node_2, components)

    new_components = dict(components)
    del new_components[node_1]
    del new_components[node_2]

    next_id = max(new_components.keys(), default=-1) + 1

    if intersection[1]:
        new_components[next_id] = {
            "dims": intersection[0],
            "pos": sorted(intersection[1]),
            "label": "I"
        }
        next_id += 1

    if residual_1[1]:
        new_components[next_id] = {
            "dims": residual_1[0],
            "pos": sorted(residual_1[1]),
            "label": "R"
        }
        next_id += 1

    if residual_2[1]:
        new_components[next_id] = {
            "dims": residual_2[0],
            "pos": sorted(residual_2[1]),
            "label": "R"
        }

    return relabel_components(new_components)'''

# ---------------------------------------------------------------------------
# InterdimConnections
# ---------------------------------------------------------------------------

class InterdimConnections2:

    def __init__(self, min_weight=0.8, n_patterns=None, save_history=False, arbitrate_scc_gather=False, keep_scc_residuals=False):
        self.min_weight = min_weight
        self.n_patterns = n_patterns
        self.save_history = save_history
        self.arbitrate_scc_gather = arbitrate_scc_gather
        self.keep_scc_residuals = keep_scc_residuals
        self._next_id = 0   

    def _new_component(self, dims, pos, label=None):
        cid = self._next_id
        self._next_id += 1
        return cid, {
            "dims": set(dims),
            "pos": sorted(pos),
            "label": label if label is not None else f"C{cid}"
        }

    def apply_split(self, node_1, node_2, components):
        intersection, residual_1, residual_2 = nodes_intersection(node_1, node_2, components)

        new_components = dict(components)

        comp_1 = components[node_1]
        comp_2 = components[node_2]

        del new_components[node_1]
        del new_components[node_2]
        new_id = self._next_id
        self._next_id += 1

        new_id = self._next_id
        self._next_id += 1
        # L'intersection prend un nouveau label
        if intersection[1]:
            new_components[new_id] = {
                "dims": intersection[0],
                "pos": sorted(intersection[1]),
                "label": f"S{new_id}"   # ou "I", mais unique c'est mieux
            }
            new_id += 1

        # Le résidu de node_1 garde le label de node_1
        if residual_1[1]:
            new_components[node_1] = {
                "dims": residual_1[0],
                "pos": sorted(residual_1[1]),
                "label": comp_1["label"]
            }

        # Le résidu de node_2 garde le label de node_2
        if residual_2[1]:
            new_components[node_2] = {
                "dims": residual_2[0],
                "pos": sorted(residual_2[1]),
                "label": comp_2["label"]
            }

        return new_components


    # ------------------------------------------------------------------
    # Graph / SCC pipeline
    # ------------------------------------------------------------------

    def compute_graph_and_scc(self, connected_components):
        initial_G = build_directed_cc_graph(connected_components, self.min_weight)
        current_components = gather_scc(
            connected_components, initial_G,
            n_patterns=self.n_patterns if self.arbitrate_scc_gather else None,
            arbitrate=self.arbitrate_scc_gather,
            keep_residuals=self.keep_scc_residuals,
        )
        current_components = relabel_components(current_components)
        coverage_scores = compute_ccp_coverage(current_components)
        coverage = compute_coverage(coverage_scores, self.n_patterns)

        final_G = build_directed_cc_graph(current_components, self.min_weight)
        sinks_list, ancestors_list = get_sink_and_ancestors_sccs(final_G, coverage_scores)

        return initial_G, final_G, current_components, coverage_scores, coverage, sinks_list, ancestors_list

    # ------------------------------------------------------------------
    # Split evaluation
    # ------------------------------------------------------------------

    '''def evaluate_split(self, node_1, node_2, current_components):
        splitted = apply_split(node_1, node_2, current_components)
        initial_G, final_G, new_components, coverage_scores, coverage, sinks_list, ancestors_list = \
            self.compute_graph_and_scc(splitted)
        return new_components, initial_G, final_G, coverage_scores, coverage, sinks_list, ancestors_list'''
    def evaluate_split_trial(self, node_1, node_2, current_components, base_coverage):
        split_components = apply_split(node_1, node_2, current_components)

        initial_G, final_G, new_components, coverage_scores, coverage, sinks_list, ancestors_list = \
            self.compute_graph_and_scc(split_components)

        return {
            "sink": node_1,
            "ancestor": node_2,
            "split_components": split_components,   # avant MergeSCC
            "components": new_components,           # après MergeSCC
            "initial_G": initial_G,
            "G": final_G,
            "coverage_scores": coverage_scores,
            "coverage": coverage,
            "gain": coverage - base_coverage,
            "sinks": sinks_list,
            "ancestors": ancestors_list,
        }


    def get_best_split(self, base_coverage, sinks_list, ancestors_list, current_components):
        best_gain = 0
        best_trial = None
        split_trials = []

        for i, node_1 in enumerate(sinks_list):
            for node_2 in ancestors_list[i]:
                trial = self.evaluate_split_trial(
                    node_1,
                    node_2,
                    current_components,
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
            best_trial["components"],
            best_trial["coverage_scores"],
            best_trial["sinks"],
            best_trial["ancestors"],
            split_trials,
        )
    
    '''def get_best_split(self, base_coverage, sinks_list, ancestors_list, current_components):
        best_gain = 0
        best_initial_G = best_G = best_components = best_scc_cov = best_sinks = best_ancestors = None
        coverage_gains_list = []

        for i, node_1 in enumerate(sinks_list):
            coverage_gains_node_1 = []
            for node_2 in ancestors_list[i]:
                new_c, new_initial_G, new_G, scc_cov, coverage_score, new_sinks, new_ancestors = \
                    self.evaluate_split(node_1, node_2, current_components)
                gain = coverage_score - base_coverage
                coverage_gains_node_1.append(gain)
                if gain > best_gain:
                    best_gain = gain
                    best_components, best_initial_G, best_G, best_scc_cov = new_c, new_initial_G, new_G, scc_cov
                    best_sinks, best_ancestors = new_sinks, new_ancestors
            coverage_gains_list.append(coverage_gains_node_1)

        return best_gain, best_initial_G, best_G, best_components, best_scc_cov, best_sinks, best_ancestors, coverage_gains_list'''

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
                    "components": self.current_components,
                    "initial_G": self.current_initial_G,
                    "G": self.current_G,
                    "coverage": self.coverage,
                    "coverage_scores": self.coverage_scores,
                    "n_patterns": n_patterns,
                })

            best_gain, best_initial_G, best_G, best_components, best_coverage_scores, best_sinks, best_ancestors, split_trials = \
                self.get_best_split(self.coverage, self.sinks_list, self.ancestors_list, self.current_components)
            if self.save_history:
                self.history[-1]["split_trials"] = split_trials
            if best_gain <= 0:
                break

            self.coverage += best_gain
            self.current_components = best_components
            self.coverage_scores = best_coverage_scores
            self.current_initial_G = best_initial_G
            self.current_G = best_G
            self.sinks_list, self.ancestors_list = best_sinks, best_ancestors
            self.n_iterations += 1

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, individual_components):
        self.individual_components = rename_initial_components(individual_components)
        self._next_id = max(self.individual_components.keys()) + 1
        self.individual_components = relabel_components(self.individual_components)
        self.current_initial_G, self.current_G, self.current_components, self.coverage_scores, self.coverage, \
            self.sinks_list, self.ancestors_list = self.compute_graph_and_scc(self.individual_components)
        self.iterative_split()
        if self.n_patterns is None:
            self.n_patterns = optimal_n_patterns(self.coverage_scores)

        effective_n = min(self.n_patterns, len(self.coverage_scores))
        top_keys = list(self.coverage_scores.keys())[:effective_n]
        self.final_components = {k: self.current_components[k] for k in top_keys}

        _, self.final_graph, _, _, _ , _ ,_ = self.compute_graph_and_scc(self.final_components)