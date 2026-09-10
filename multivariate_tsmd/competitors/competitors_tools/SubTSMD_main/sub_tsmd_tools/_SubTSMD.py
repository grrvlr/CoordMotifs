import numpy as np
import pandas as pd
from locomotif import locomotif

from ._sub_tsmd import apply_sub_tsmd
from ._SubspaceMotifSet import SubspaceMotifSet


class SubTSMD:
    """
    A class to apply SubTSMD on a time series, in which the motifs in
    each attribute are discovered using LoCoMotif.
    """

    # SubTSMD parameters
    delta: float
    linkage: str
    # LoCoMotif parameters
    l_min: int
    l_max: int
    rho: float | None
    max_number_motif_sets: int | None
    warping: bool
    overlap: float

    def __init__(
        self,
        l_min: int,
        l_max: int,
        delta: float = 0.5,
        linkage: str = "average",
        rho: float = None,
        max_number_motif_sets: float = None,
        warping: bool = True,
        overlap: float = 0.0,
    ):
        assert (
            0.5 <= delta <= 1
        ), f"Invalid delta given: {delta}, must be in the interval [0.5, 1]."
        assert linkage in [
            "complete",
            "average",
        ], f"Invalid linkage method given: '{linkage}', valid options are 'complete' or 'average'."
        assert l_min > 0, f"Invalid l_min given: {l_min}, should be larger than 0. "
        assert (
            l_max >= l_min
        ), f"Invalid l_max given: {l_max}, should be equal to or larger than l_min (={l_min})."
        assert (
            rho is None or 0 <= rho <= 1
        ), f"Invalid rho given: {rho}, should be None or in the interval [0, 1]."
        assert (
            max_number_motif_sets is None or max_number_motif_sets > 0
        ), f"Invalid max_number_motif_sets given: {max_number_motif_sets}, should be None or larger than 0."

        self.delta = delta
        self.linkage = linkage
        self.l_min = l_min
        self.l_max = l_max
        self.rho = rho
        self.max_number_motif_sets = max_number_motif_sets
        self.warping = warping
        self.overlap = overlap

    def apply(
        self,
        X: np.ndarray | pd.DataFrame,
        inclusion_constraint: list[str] | list[int] = None,
        exclusion_constraint: list[str] | list[int] = None,
        size_constraint: int = 1,
        co_occurrence_constraint: list[list[str] | list[int]] = None,
        start_mask: np.array = None,
        end_mask: np.array = None,
    ) -> list[SubspaceMotifSet]:
        # Mine motifs in the attributes independently using LoCoMotif
        independent_motif_sets = []
        all_masks = _get_motif_discovery_masks(
            X=X,
            inclusion_constraint=inclusion_constraint,
            exclusion_constraint=exclusion_constraint,
            co_occurrence_constraint=co_occurrence_constraint,
        )
        for mask in all_masks:
            # Discover the motif sets
            motif_sets = locomotif.apply_locomotif(
                ts=_z_normalize(X[:, mask]),
                rho=self.rho,
                l_min=self.l_min,
                l_max=self.l_max,
                nb=self.max_number_motif_sets,
                warping=self.warping,
                overlap=self.overlap,
                start_mask=None if start_mask is None else start_mask.copy(),
                end_mask=None if end_mask is None else end_mask.copy(),
            )

            # Format the motif sets
            independent_motif_sets.append(
                [
                    (
                        mask,
                        np.repeat(
                            np.array(motif_set).reshape(len(motif_set), 2, 1),
                            mask.sum(),
                            axis=-1,
                        ),
                    )
                    for (_, motif_set) in motif_sets
                ]
            )

        # Replace the indices from LoCoMotif by time indices
        independent_motif_sets = _replace_indices_by_time_index(
            independent_motif_sets, X
        )

        # Apply SubTSMD
        inclusion_constraint_set = (
            inclusion_constraint is not None and len(inclusion_constraint) >= 1
        )
        subspace_motif_sets = apply_sub_tsmd(
            independent_motif_sets,
            delta=self.delta,
            linkage=self.linkage,
            inclusion_constraint_set=inclusion_constraint_set,
        )

        # Return the motif sets that satisfy the size constraint
        return [
            SubspaceMotifSet.from_arrays(motif_set)
            for motif_set in subspace_motif_sets
            if motif_set[0].sum() >= size_constraint
        ]


def _z_normalize(X: np.array) -> np.array:
    return (X - X.mean(axis=0)) / X.std(axis=0)


def _get_motif_discovery_masks(
    X: np.ndarray | pd.DataFrame,
    inclusion_constraint: list[str] | list[int] = None,
    exclusion_constraint: list[str] | list[int] = None,
    co_occurrence_constraint: list[list[str] | list[int]] = None,
) -> list[np.ndarray]:
    used_attributes = np.zeros(shape=X.shape[1], dtype=bool)
    masks = []

    def all_of_type(to_check, of_type):
        return all(map(lambda x: isinstance(x, of_type), to_check))

    def translate_to_integers(constraint_to_translate, columns_list):
        if all_of_type(constraint_to_translate, int):
            return [columns_list.index(col) for col in constraint_to_translate]
        else:
            return constraint_to_translate

    all_constraints = []
    if inclusion_constraint is not None:
        all_constraints.append(inclusion_constraint)
    if exclusion_constraint is not None:
        all_constraints.append(exclusion_constraint)
    if co_occurrence_constraint is not None:
        all_constraints.extend(co_occurrence_constraint)

    if isinstance(X, pd.DataFrame):
        # Check if constraints are correctly formatted
        for constraint in all_constraints:
            all_string = all_of_type(constraint, str)
            all_int = all_of_type(constraint, int)
            if not (all_string or all_int):
                raise ValueError(
                    f"If a pandas dataframe is given, the constraints should be either all integers or all strings! Received: {constraint}"
                )
            if all_string and any(value not in X.columns for value in constraint):
                raise ValueError(
                    f"If a pandas dataframe is given, all string constraints should be a column! Received: {constraint}"
                )
            if all_int and max(constraint) >= X.shape[1]:
                raise ValueError(
                    f"If a pandas dataframe is given, the attribute constraint index must be smaller than {X.shape[1]} (zero-indexed)! Received: {constraint}"
                )

        # Translate the constraints to integers
        columns = list(X.columns)
        if inclusion_constraint is not None:
            inclusion_constraint = translate_to_integers(inclusion_constraint, columns)
        if exclusion_constraint is not None:
            exclusion_constraint = translate_to_integers(exclusion_constraint, columns)
        if co_occurrence_constraint is not None:
            co_occurrence_constraint = [
                translate_to_integers(constraint, columns)
                for constraint in co_occurrence_constraint
            ]

    else:  # X is a numpy array
        for constraint in all_constraints:
            if not all_of_type(constraint, int):
                raise ValueError(
                    f"If a numpy array is given, the constraints should exist of only integers! Received: {constraint}"
                )
            if max(constraint) >= X.shape[1]:
                raise ValueError(
                    f"If a numpy array is given, the attribute constraint index must be smaller than {X.shape[1]} (zero-indexed)! Received: {constraint}"
                )

    if inclusion_constraint is not None:
        new_mask = np.zeros_like(used_attributes)
        new_mask[inclusion_constraint] = True
        masks.append(new_mask)

    if exclusion_constraint is not None:
        used_attributes[exclusion_constraint] = True

    if co_occurrence_constraint is not None:
        for constraint in co_occurrence_constraint:
            new_mask = np.zeros_like(used_attributes)
            new_mask[constraint] = True
            masks.append(new_mask)
            used_attributes[constraint] = True

    # Add the remaining attributes
    for i, used in enumerate(used_attributes):
        if not used:
            new_mask = np.zeros_like(used_attributes)
            new_mask[i] = True
            masks.append(new_mask)

    # Return the masks
    return masks


def _replace_indices_by_time_index(
    independent_motif_sets: list[list[(np.array, np.ndarray)]],
    X: np.ndarray | pd.DataFrame,
) -> list[list[(np.array, np.ndarray)]]:
    if isinstance(X, pd.DataFrame):
        return [
            [
                (
                    mask,
                    X.index.values[motif_set],
                )  # Reindex based on the index of the dataframe
                for mask, motif_set in motif_sets
            ]
            for motif_sets in independent_motif_sets
        ]

    else:
        return independent_motif_sets
