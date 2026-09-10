import functools

import numba as nb
import numpy as np

type_motif_set = nb.types.Tuple(
    (
        nb.types.Array(nb.types.boolean, 1, "C"),  # The mask
        nb.types.Array(nb.types.float32, 3, "C"),  # The motif locations
    )
)


def apply_sub_tsmd(
    independent_motif_sets: list[list[(np.array, np.ndarray)]],
    delta: float = 0.5,
    linkage: str = "average",
    inclusion_constraint_set: bool = False,
) -> list[(np.array, np.ndarray)]:
    # Filter empty motif sets
    independent_motif_sets = list(filter(lambda s: len(s) > 0, independent_motif_sets))

    # Nothing to do if there are no motif sets
    if len(independent_motif_sets) == 0:
        return []

    # Set the type of the motifs
    independent_motif_sets = [
        [
            (
                independent_motif_sets[i][j][0].astype(np.bool_),
                independent_motif_sets[i][j][1].astype(np.float32),
            )
            for j in range(len(independent_motif_sets[i]))
        ]
        for i in range(len(independent_motif_sets))
    ]

    # Iteratively merge the motif sets
    return functools.reduce(
        lambda x, y: _merge(
            x,
            y,
            delta=delta,
            linkage=linkage,
            inclusion_constraint_set=inclusion_constraint_set,
        ),
        independent_motif_sets,
    )


@nb.njit(
    nb.float32(nb.float32, nb.float32, nb.float32, nb.float32),
    fastmath=True,
    cache=True,
)
def _overlap_rate(start_1, end_1, start_2, end_2):
    return max(
        0,
        (min(end_1, end_2) - max(start_1, start_2))
        / (max(end_1, end_2) - min(start_1, start_2)),
    )


@nb.njit(
    nb.types.Tuple((type_motif_set, type_motif_set, type_motif_set))(
        type_motif_set, type_motif_set, nb.float32, nb.types.unicode_type
    ),
    fastmath=True,
    cache=True,
)
def _match(motif_set, motif_set_prime, delta, linkage):

    matches = np.full(shape=motif_set[1].shape[0], fill_value=-1)
    matches_prime = np.full(shape=motif_set_prime[1].shape[0], fill_value=-1)

    for i in range(motif_set[1].shape[0]):
        for j in range(motif_set_prime[1].shape[0]):

            # We can skip this due to the property
            if matches_prime[j] != -1:
                continue

            # Between the overlap rates across each attribute
            overlap_rates = np.ones(
                shape=(motif_set[1][i].shape[1], motif_set_prime[1][j].shape[1])
            )
            for a in range(motif_set[1][i].shape[1]):
                for b in range(motif_set_prime[1][j].shape[1]):
                    overlap_rates[a, b] = _overlap_rate(
                        start_1=motif_set[1][i, 0, a],
                        end_1=motif_set[1][i, 1, a],
                        start_2=motif_set_prime[1][j, 0, b],
                        end_2=motif_set_prime[1][j, 1, b],
                    )

            # The overlap rate between subspace motifs
            if linkage == "complete":
                overlap = (
                    overlap_rates.min()
                )  # min instead of max because overlap rate is a similarity and not a distance
            elif linkage == "average":
                overlap = overlap_rates.mean()
            else:
                raise ValueError(
                    f"Unknown linkage method: '{linkage}'. Expected one of 'complete' or 'average'."
                )

            # Check if there is a match
            if overlap > delta or (
                delta == 1 and overlap == 1
            ):  # '>' and not '>=', to ensure single motif overlaps
                matches[i] = j
                matches_prime[j] = i
                break  # We can break the inner for-loop due to the property

    # Combine the matched motifs
    matched_mask = motif_set[0] | motif_set_prime[0]
    matched_motifs = np.zeros(
        shape=(np.sum(matches_prime != -1), 2, matched_mask.shape[0]), dtype=nb.float32
    )
    matched_motifs[:, :, motif_set[0]] = motif_set[1][matches != -1]
    matched_motifs[:, :, motif_set_prime[0]] = motif_set_prime[1][
        matches[matches != -1]
    ]
    matched_motifs = matched_motifs[:, :, matched_mask]

    # Extract the motifs that were not matched
    unmatched_motifs = (motif_set[0], motif_set[1][matches == -1])
    unmatched_motifs_prime = (
        motif_set_prime[0],
        motif_set_prime[1][matches_prime == -1],
    )

    # Return all the motifs
    return (matched_mask, matched_motifs), unmatched_motifs, unmatched_motifs_prime


@nb.njit(fastmath=True, cache=True)
def _merge(motif_sets, motif_sets_prime, delta, linkage, inclusion_constraint_set):

    merged_motif_sets = []

    for i in range(len(motif_sets)):
        for j in range(len(motif_sets_prime)):

            # Match the motifs (and also obtain the unmatched motifs)
            matched_motifs, unmatched_motifs, unmatched_motifs_prime = _match(
                motif_sets[i], motif_sets_prime[j], delta=delta, linkage=linkage
            )

            if matched_motifs[1].shape[0] >= 2:
                # Add the matched motif sets
                merged_motif_sets.append(matched_motifs)

                # Update the motif sets that have not been matched
                motif_sets[i] = unmatched_motifs
                motif_sets_prime[j] = unmatched_motifs_prime

                # Discard if only one motif set remains
                if unmatched_motifs[1].shape[0] <= 1:
                    print()
                    motif_sets[i] = (
                        np.zeros(shape=matched_motifs[0].shape[0], dtype=nb.bool_),
                        np.empty(shape=(0, 2, 0), dtype=nb.float32),
                    )
                if unmatched_motifs_prime[1].shape[0] <= 1:
                    motif_sets_prime[j] = (
                        np.zeros(shape=matched_motifs[0].shape[0], dtype=nb.bool_),
                        np.empty(shape=(0, 2, 0), dtype=nb.float32),
                    )

    # Add all the motif sets
    merged_motif_sets += [
        motif_set for motif_set in motif_sets if motif_set[1].shape[0] > 0
    ]
    if not inclusion_constraint_set:
        # Only include the second list if there was no inclusion constraint set.
        merged_motif_sets += [
            motif_set_prime
            for motif_set_prime in motif_sets_prime
            if motif_set_prime[1].shape[0] > 0
        ]

    # Return all the motif sets
    return merged_motif_sets
