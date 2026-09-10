import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from joblib import Parallel, delayed

from src.tools.basic_persistence import BasicPersistence, MDThresholdPersistenceMST
from src.tools.threshold import otsu_jump
from src.tools.neighborhood import KNN
from src.tools.interdim_connections import InterdimConnections
from src.tools.plots import plot_directed_cc_graph_scc_roles, plot_candidates


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def process_dim(i, n_neighbors, knn_wlen, distance_name, signal):
    knn = KNN(n_neighbors, knn_wlen, distance_name, n_jobs=1, time_connection=True)
    knn.fit(signal[:, i])
    base_persistence_ = BasicPersistence()
    base_persistence_.fit(knn.filtration_)
    return base_persistence_, knn.filtration_

def _stack_with_dim(arrays):
    parts = []
    for dim, arr in enumerate(arrays):
        if arr is not None and len(arr) > 0:
            dim_col = np.full((arr.shape[0], 1), dim, dtype=int)
            parts.append(np.hstack((dim_col, arr)))
    return np.vstack(parts) if parts else None


# ---------------------------------------------------------------------------
# Rank / motif utilities
# ---------------------------------------------------------------------------

def compute_global_ranks(candidate_sets, ranks):
    """Compute average rank per (pos, comp) pair, sorted ascending."""
    global_ranks = {}
    for comp_id, (dims, positions) in candidate_sets.items():
        for pos in positions:
            pos_ranks = [ranks[(dim, pos)] for dim in dims if (dim, pos) in ranks]
            if pos_ranks:
                global_ranks[(pos, comp_id)] = np.mean(pos_ranks)
    return dict(sorted(global_ranks.items(), key=lambda x: x[1]))


# ---------------------------------------------------------------------------
# CoordMotifs
# ---------------------------------------------------------------------------

class CoordMotifs:

    def __init__(
        self,
        wlen: int,
        n_patterns=None,
        n_neighbors=5,
        jump=2,
        min_wlen=None,
        max_wlen=None,
        distance_name_for_persistence="LTNormalizedEuclidean",
        alpha=10,
        beta=0,
        min_connection_weight=0.8,
        individual_birth_cut=True,
        n_jobs=1,
        use_ktanh=True,
        save_history=False,
        noise_threshold=0.4,
    ):
        self.wlen = wlen
        self.n_neighbors = n_neighbors
        self.n_patterns = n_patterns
        self.jump = jump
        self.distance_name = distance_name_for_persistence
        self.min_wlen = min_wlen if min_wlen is not None else wlen
        self.max_wlen = max_wlen if max_wlen is not None else wlen
        self.alpha = alpha
        self.beta = beta
        self.min_connection_weight = min_connection_weight
        self.individual_birth_cut = individual_birth_cut
        self.n_jobs = n_jobs
        self.use_ktanh = use_ktanh
        self.save_history = save_history
        self.noise_threshold = noise_threshold

        self.p_cut_ = None
        self.b_cut_ = None
        self.timings_ = {}

    # ------------------------------------------------------------------
    # Distance transform
    # ------------------------------------------------------------------

    def _ktanh(self, X: np.ndarray, alpha=None, beta=None) -> np.ndarray:
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta
        norm_factor = np.tanh(beta ** 2 * alpha) - np.tanh(-alpha * (4 - beta ** 2))
        dists = np.tanh(beta ** 2 * alpha) - np.tanh(-alpha * (X ** 2 - beta ** 2))
        return 2 * np.sqrt(dists / norm_factor)

    # ------------------------------------------------------------------
    # Persistence pipeline
    # ------------------------------------------------------------------

    def _base_persistence(self) -> None:
        self.n, self.n_dims = self.signal.shape
        results = Parallel(n_jobs=min(self.n_jobs, self.n_dims), prefer="processes")(
            delayed(process_dim)(i, self.n_neighbors, self.wlen, self.distance_name, self.signal)
            for i in range(self.n_dims)
        )
        self.persistences_ = [r[0].get_persistence() for r in results]
        self.filtrations_ = [r[1] for r in results]

        self.persistence_array_ = _stack_with_dim(self.persistences_)
        self.filtration_array_ = _stack_with_dim(self.filtrations_)
        self.filtration_array_raw_ = self.filtration_array_.copy()

        if self.persistence_array_ is not None:
            self.persistence_array_ = self.persistence_array_[
                np.lexsort((self.persistence_array_[:, 0], self.persistence_array_[:, -2]))
            ]
        if self.filtration_array_ is not None:
            self.filtration_array_ = self.filtration_array_[
                np.lexsort((self.filtration_array_[:, 0], self.filtration_array_[:, -1]))
            ]

    def _thresholds(self) -> None:
        pers = self.get_persistence(True)
        filtered_pers = pers[pers[:, 2] - pers[:, 1] > 0]
        self.p_cut_, self.b_cut_ = otsu_jump(filtered_pers[:-self.n_dims, 1:3], jump=self.jump)

    def _persistence_with_thresholds(self) -> None:
        self.tpmst = MDThresholdPersistenceMST(
            persistence_threshold=self.p_cut_,
            birth_threshold=self.b_cut_,
        )
        filtration = self.filtration_array_raw_.copy()
        filtration[:, -1] = self._ktanh(filtration[:, -1])
        self.tpmst.fit(filtration)
        
    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process(self) -> None:
        self.univariate_candidates = self.tpmst.components
        self.connect_dims = InterdimConnections(min_weight=self.min_connection_weight,
            n_patterns=self.n_patterns,
            save_history=self.save_history
        )
        self.connect_dims.fit(self.univariate_candidates)
        self.final_candidates = self.connect_dims.final_candidates
        self.global_ranks = compute_global_ranks(self.final_candidates, self.tpmst.rank_)
        self.overlap_mask_, self.prediction_mask_, self.prediction_dimension_ = self.get_final_motifs()

    # ------------------------------------------------------------------
    # Public fit / setters
    # ------------------------------------------------------------------

    def fit(self, signal: np.ndarray) -> None:
        self.signal = signal
        self._base_persistence()
        self._thresholds()
        self._persistence_with_thresholds()
        self._post_process()

    def set_distance_params(self, alpha=None, beta=None):
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        self._thresholds()
        self._persistence_with_thresholds()
        self._post_process()

    def set_cut_values(self, p_cut=None, b_cut=None):
        if p_cut is not None:
            self.p_cut_ = p_cut
        if b_cut is not None:
            self.b_cut_ = b_cut
        self._persistence_with_thresholds()
        self._post_process()

    def set_min_connection_weight(self, weight):
        self.min_connection_weight = weight
        self._post_process()
    # ------------------------------------------------------------------
    # Persistence accessors
    # ------------------------------------------------------------------

    def get_persistence(self, with_infinite_point=True) -> np.ndarray:
        pers = np.copy(self.persistence_array_)
        if with_infinite_point:
            pers[-self.n_dims:, 2] = np.max(pers[:, 2])
        if self.use_ktanh:
            pers[:, 1:3] = self._ktanh(pers[:, 1:3])
        return pers

    def get_individual_persistence(self, with_infinite_point=True):
        pers_list = [pers.copy() for pers in self.persistences_]
        for pers in pers_list:
            if pers is None or len(pers) == 0:
                continue
            if with_infinite_point:
                pers[-1, 2] = np.max(pers[:, 2])
            if self.use_ktanh:
                pers[:, 1:3] = self._ktanh(pers[:, 1:3])
        return pers_list

    # ------------------------------------------------------------------
    # Motif extraction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Segment scoring (used by get_final_motifs)
    # ------------------------------------------------------------------

    def _get_scored_segments(self, ratio=0.5):
        """
        Pour chaque comp, segmente les positions en runs contigus,
        score chaque run (min(length, max_wlen) * n_dims),
        filtre intra-motif (garder >= ratio * max_score),
        et retourne tous les segments triés par score décroissant.
        """
        all_segments = []

        for comp, (dims_set, positions) in self.final_candidates.items():
            positions = sorted(positions)
            n_dims = len(dims_set)
            if not positions:
                continue

            current = [positions[0]]
            for p in positions[1:]:
                if p == current[-1] + 1:
                    current.append(p)
                else:
                    all_segments.append(self._build_segment(current, comp, n_dims))
                    current = [p]
            all_segments.append(self._build_segment(current, comp, n_dims))

        by_comp = {}
        for seg in all_segments:
            by_comp.setdefault(seg["comp"], []).append(seg)

        filtered = []
        for comp, segs in by_comp.items():
            if not segs:
                continue
            max_score = max(s["score"] for s in segs)
            kept = [s for s in segs if s["score"] >= ratio * max_score]
            if not kept:
                kept = [max(segs, key=lambda x: x["score"])]
            filtered.extend(kept)

        filtered.sort(key=lambda x: x["score"], reverse=True)
        return filtered

    def _build_segment(self, positions, comp, n_dims):
        length = len(positions)
        return {
            "comp": comp,
            "start": positions[0],
            "end": positions[-1],
            "length": length,
            "n_dims": n_dims,
            "score": min(length, self.max_wlen) * n_dims,
            "positions": positions,
        }

    # ------------------------------------------------------------------
    # Overlap validity checks (used by extend_occurrence)
    # ------------------------------------------------------------------

    def _is_valid_left(self, pos, overlap_mask, comp_dims):
        if pos <= 0:
            return False
        if any(overlap_mask[pos - 1 : pos + 1, dim].any() for dim in comp_dims):
            return False
        return True

    def _is_valid_right(self, pos, overlap_mask, comp_dims, current_wlen=None):
        wlen = current_wlen if current_wlen is not None else self.min_wlen
        if pos + wlen >= self.n:
            return False
        if any(overlap_mask[pos + wlen : pos + wlen + 2, dim].any() for dim in comp_dims):
            return False
        return True

    # ------------------------------------------------------------------
    # Occurrence extension
    # ------------------------------------------------------------------

    def _extend_occurrence(self, occ, overlap_mask, comp_dims):
        """
        Étend une occurrence d'un point à la fois (gauche ou droite),
        en choisissant le côté avec le meilleur global_rank.
        Met à jour overlap_mask et prediction_mask à chaque pas.
        """
        comp = occ["comp"]

        while occ["length"] < self.max_wlen and occ["active"]:
            left_pos = occ["left_pos"]
            right_pos = occ["right_pos"]

            left_valid = self._is_valid_left(left_pos, overlap_mask, comp_dims)
            right_valid = self._is_valid_right(right_pos, overlap_mask, comp_dims, occ["length"])

            left_score = self.global_ranks.get((left_pos, comp), np.inf) if left_valid else np.inf
            right_score = self.global_ranks.get((right_pos, comp), np.inf) if right_valid else np.inf

            if left_score == np.inf and right_score == np.inf:
                occ["active"] = False
                break

            if left_score <= right_score:
                # le point ajouté est left_pos (début de fenêtre = point physique)
                new_pos = left_pos
                occ["segment"] = (new_pos, occ["segment"][1])
                occ["left_pos"] = new_pos - 1
                overlap_mask[new_pos, list(comp_dims)] = True
            else:
                # right_pos est un début de fenêtre, le point physique ajouté est le end courant
                new_pos = occ["segment"][1]
                occ["segment"] = (occ["segment"][0], new_pos + 1)
                occ["right_pos"] = right_pos + 1
                overlap_mask[new_pos, list(comp_dims)] = True

            occ["length"] += 1

            # recalcul de best_score pour le tri global
            lv = self._is_valid_left(occ["left_pos"], overlap_mask, comp_dims)
            rv = self._is_valid_right(occ["right_pos"], overlap_mask, comp_dims, occ["length"])
            new_left = self.global_ranks.get((occ["left_pos"], comp), np.inf) if lv else np.inf
            new_right = self.global_ranks.get((occ["right_pos"], comp), np.inf) if rv else np.inf
            occ["best_score"] = min(new_left, new_right)

            if occ["length"] >= self.max_wlen:
                occ["active"] = False

        return occ, overlap_mask

    def _extend_all_occurrences(self, segment_occ, overlap_mask):
        """
        Étend toutes les occurrences actives dans l'ordre de best_score croissant.
        """
        while True:
            active = {k: occ for k, occ in segment_occ.items() if occ["active"]}
            if not active:
                break

            key = min(active, key=lambda k: active[k]["best_score"])
            occ = active[key]
            comp_dims, _ = self.final_candidates[occ["comp"]]

            occ, overlap_mask = self._extend_occurrence(occ, overlap_mask, comp_dims)
            segment_occ[key] = occ

        return segment_occ, overlap_mask

    # ------------------------------------------------------------------
    # Main motif extraction
    # ------------------------------------------------------------------

    def get_final_motifs(self):

        self.effective_n_motifs = min(self.connect_dims.n_patterns, len(self.final_candidates))

        overlap_mask = np.zeros((self.n, self.n_dims), dtype=bool)
        prediction_mask = np.zeros((self.effective_n_motifs, self.n), dtype=bool)
        prediction_dimension = [list(dims) for _, (dims, _) in self.final_candidates.items()]

        segment_occ = {}
        segments = self._get_scored_segments(ratio=self.noise_threshold)

        # === 1. placement initial (min_wlen) ===
        for seg_id, seg in enumerate(segments):
            comp = seg["comp"]
            comp_dims, _ = self.final_candidates[comp]

            # tri local des points par global_rank
            candidates = sorted(
                seg["positions"],
                key=lambda pos: self.global_ranks.get((pos, comp), np.inf),
            )

            chosen_pos = None
            for pos in candidates:
                end = pos + self.min_wlen
                if end > self.n:
                    continue

                # buffer d'un point de chaque côté pour garantir le gap
                buf_start = max(0, pos - 1)
                buf_end = min(self.n, end + 1)
                has_overlap = any(
                    overlap_mask[buf_start:buf_end, dim].any() for dim in comp_dims
                )
                if not has_overlap:
                    chosen_pos = pos
                    break

            if chosen_pos is None:
                continue

            start = chosen_pos
            end = start + self.min_wlen
            key = (comp, seg_id)

            # marquage overlap avec buffer + prediction
            buf_start = max(0, start - 1)
            buf_end = min(self.n, end + 1)
            for dim in comp_dims:
                overlap_mask[buf_start:buf_end, dim] = True
            prediction_mask[comp, start:end] = True

            if self.min_wlen == self.max_wlen:
                segment_occ[key] = {
                    "segment": (start, end),
                    "best_score": np.inf,
                    "active": False,
                    "comp": comp,
                }
            else:
                left_pos = start - 1
                right_pos = start + 1

                lv = self._is_valid_left(left_pos, overlap_mask, comp_dims)
                rv = self._is_valid_right(right_pos, overlap_mask, comp_dims)

                left_score = self.global_ranks.get((left_pos, comp), np.inf) if lv else np.inf
                right_score = self.global_ranks.get((right_pos, comp), np.inf) if rv else np.inf

                segment_occ[key] = {
                    "segment": (start, end),
                    "left_pos": left_pos,
                    "right_pos": right_pos,
                    "left_score": left_score,
                    "right_score": right_score,
                    "best_score": min(left_score, right_score),
                    "length": self.min_wlen,
                    "comp": comp,
                    "active": min(left_score, right_score) != np.inf,
                }

        # === 2. extension vers max_wlen ===
        if self.min_wlen < self.max_wlen:
            segment_occ, overlap_mask = self._extend_all_occurrences(segment_occ, overlap_mask)

        self.segment_occ_ = segment_occ

        # === 3. reconstruction de prediction_mask depuis segment_occ ===
        prediction_mask = np.zeros((self.effective_n_motifs, self.n), dtype=int)
        for (comp, _), occ in segment_occ.items():
            start, end = occ["segment"]
            prediction_mask[comp, start:end] = 1

        # === 4. nettoyage ===
        non_empty = [comp for comp in range(self.effective_n_motifs) if prediction_mask[comp].any()]
        prediction_mask = prediction_mask[non_empty]
        prediction_dimension = [prediction_dimension[comp] for comp in non_empty]

        return overlap_mask, prediction_mask, prediction_dimension

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_persistence_diagram(self):
        pers = self.get_persistence(True)[:, 1:3]
        pers = pers[pers[:, 1] - pers[:, 0] != 0]
        mask1 = pers[:, 0] > self.b_cut_
        mask1 += (pers[:, 0] <= self.b_cut_) * ((pers[:, 1] - pers[:, 0]) <= self.p_cut_)
        mask2 = (pers[:, 0] <= self.b_cut_) * ((pers[:, 1] - pers[:, 0]) > self.p_cut_)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.add_patch(Polygon([[0, 0], [2, 0], [2, 2]], color="grey", alpha=0.25))
        for v in [0, 2]:
            ax.hlines(v, 0, 2, color="black", lw=0.5, zorder=1)
            ax.vlines(v, 0, 2, color="black", lw=0.5, zorder=1)
        ax.scatter(*pers[mask1].T, color="tab:blue", zorder=2)
        ax.scatter(*pers[mask2].T, color="tab:orange", zorder=2)
        ax.vlines(self.b_cut_, 0, 2, color="red", zorder=3)
        ax.add_patch(Polygon([[0, self.p_cut_], [2 - self.p_cut_, 2]], color="red", zorder=3))
        fig.tight_layout()
        plt.show()

    def plot_individual_candidates(self):
        plot_candidates(self.univariate_candidates, self.n, self.n_dims)

    def plot_final_candidates(self):
        plot_candidates(self.final_candidates, self.n, self.n_dims)

    def plot_history(self):
        if not self.connect_dims.history:
            print("No history available to plot.")
            return
        print(f"Initial candidates:")
        plot_candidates(self.univariate_candidates, self.n, self.n_dims)
        for time_frame in self.connect_dims.history:
            print(f"Coverage: {time_frame['coverage']}")
            plot_directed_cc_graph_scc_roles(time_frame["initial_G"])
            plot_directed_cc_graph_scc_roles(time_frame["G"])
            plot_candidates(time_frame["candidates"], self.n, self.n_dims)
        print(f"Final candidates:")
        plot_candidates(self.final_candidates, self.n, self.n_dims)