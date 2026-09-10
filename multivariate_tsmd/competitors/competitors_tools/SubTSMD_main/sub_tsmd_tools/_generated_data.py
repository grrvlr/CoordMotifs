import numpy as np
import pandas as pd
import scipy


def _generate_unique_sets(N, K, f, rng):
    elements = list(range(N))  # Assuming elements are 0 to N-1
    unique_sets = set()

    while len(unique_sets) < f:
        new_set = tuple(sorted(rng.choice(elements, K)))
        unique_sets.add(new_set)

    return list(unique_sets)


def _select_values(N, D, K):
    # if D > N:
    #     raise ValueError("D cannot be larger than N")
    # if K * D < N:
    #     raise ValueError("K * D must be at least N to cover all elements")

    result = np.empty((K, D), dtype=int)
    all_indices = np.arange(N)

    # Start with an empty set of selected values
    selected = set()
    for i in range(K):
        # Try to prioritize values not yet used
        remaining = list(set(all_indices) - selected)

        if len(remaining) >= D:
            chosen = np.random.choice(remaining, size=D, replace=False)
        else:
            # Choose all remaining ones first
            chosen = list(remaining)
            needed = D - len(chosen)
            # Fill the rest randomly from already seen values
            fill_choices = list(selected)
            chosen += list(np.random.choice(fill_choices, size=needed, replace=False))
            chosen = np.array(chosen)

        result[i] = chosen
        selected.update(chosen)

    return result


def generate(
    dimension: int,
    nb_motif_sets,
    min_motif_set_size,
    max_motif_set_size,
    min_motif_dimension,
    max_motif_dimension,
    univariate_motifs,
    min_motif_length,
    max_motif_length,
    white_space,
    nb_motif_repositions,
    noise_general,
    noise_non_motif,
    seed,
):
    # Initial setup
    rng = np.random.default_rng(seed)
    time_series_length = (
        max_motif_set_size * max_motif_length * nb_motif_sets + white_space
    ) * 2  # Double to be sure, will be removed anyway

    # Create the positions of the motif sets
    motif_sets = []
    available_positions = np.ones(shape=(time_series_length, dimension), dtype=bool)
    for _ in range(nb_motif_sets):

        motif_set_found = False
        while not motif_set_found:

            motif_set_subspace_size = rng.integers(
                min_motif_dimension, max_motif_dimension + 1
            )
            motif_set_size = rng.integers(min_motif_set_size, max_motif_set_size + 1)

            mask = np.zeros(dimension, dtype=bool)
            mask[rng.choice(dimension, motif_set_subspace_size, replace=False)] = True

            motif_set_found = True
            locations = np.empty(
                shape=(motif_set_size, 2, motif_set_subspace_size), dtype=int
            )
            updated_available_positions = available_positions.copy()
            for i in range(motif_set_size):
                # Find the length for the motif across all dimensions
                motif_lengths = rng.integers(
                    min_motif_length, max_motif_length + 1, size=dimension
                )
                max_length = np.max(motif_lengths[mask])

                # Find all available start positions, and select a random one
                possible_start_positions = []
                for start in range(time_series_length - max_length):
                    if np.all(
                        updated_available_positions[start : start + max_length, mask]
                    ):
                        possible_start_positions.append(start)
                if len(possible_start_positions) == 0:
                    motif_set_found = False
                    break
                start_position = rng.choice(possible_start_positions)

                # Let all the motifs start at the same position
                c = -1
                for d in range(dimension):
                    if not mask[d]:
                        continue
                    c += 1
                    locations[i, 0, c] = start_position + rng.integers(
                        max_length - motif_lengths[d] + 1
                    )
                    locations[i, 1, c] = locations[i, 0, c] + motif_lengths[d]

                # Mark the positions as unavailable
                for d in range(locations.shape[2]):
                    updated_available_positions[
                        locations[i, 0, d] : locations[i, 1, d], mask
                    ] = False

        # Update the available positions
        available_positions = updated_available_positions.copy()

        # Save the motif set
        motif_sets.append((mask, locations))

        # Remove white spaces
    white_spaces = np.all(available_positions, axis=1)
    if white_spaces.sum() > white_space:
        # Find positions to remove
        to_remove = np.zeros(shape=time_series_length, dtype=bool)
        to_remove[
            rng.choice(
                np.where(white_spaces)[0],
                white_spaces.sum() - white_space,
                replace=False,
            )
        ] = True

        # Update the time series length
        time_series_length -= to_remove.sum()

        # Update the motif sets to take the removed positions into account
        cum_sum = np.cumsum(to_remove)
        for _, motif_set in motif_sets:
            motif_set -= cum_sum[motif_set]

            # Recompute the available positions
    available_positions = np.ones(shape=(time_series_length, dimension), dtype=bool)
    for mask, motif_set in motif_sets:
        for motif in motif_set:
            c = -1
            for d in range(dimension):
                if not mask[d]:
                    continue
                c += 1
                available_positions[motif[0, c] : motif[1, c], d] = False

    for _ in range(nb_motif_repositions):
        for j, (mask, motif_set) in enumerate(motif_sets):
            for k, motif in enumerate(motif_set):

                zero_based_motif = motif - motif.min()

                c = -1
                for d in range(dimension):
                    if not mask[d]:
                        continue
                    c += 1
                    available_positions[motif[0, c] : motif[1, c], d] = True

                start_position = np.zeros(shape=time_series_length, dtype=bool)
                for t in range(time_series_length - zero_based_motif.max()):
                    start_position[t] = True

                    c = -1
                    for d in range(dimension):
                        if not mask[d]:
                            continue
                        c += 1
                        if not available_positions[
                            t + zero_based_motif[0, c] : t + zero_based_motif[1, c], d
                        ].all():
                            start_position[t] = False
                            break

                new_motif = zero_based_motif + rng.choice(np.where(start_position)[0])
                c = -1
                for d in range(dimension):
                    if not mask[d]:
                        continue
                    c += 1
                    available_positions[new_motif[0, c] : new_motif[1, c], d] = False
                motif_sets[j][1][k] = new_motif

                # Remove white spaces (again)
    white_spaces = np.all(available_positions, axis=1)
    if white_spaces.sum() > white_space:
        # Find positions to remove
        to_remove = np.zeros(shape=time_series_length, dtype=bool)
        to_remove[
            rng.choice(
                np.where(white_spaces)[0],
                white_spaces.sum() - white_space,
                replace=False,
            )
        ] = True

        # Update the time series length
        time_series_length -= to_remove.sum()

        available_positions = available_positions[~to_remove, :]

        # Update the motif sets to take the removed positions into account
        cum_sum = np.cumsum(to_remove)
        for _, motif_set in motif_sets:
            motif_set -= cum_sum[motif_set]

    # Create X
    X = np.zeros(shape=(time_series_length, dimension))
    # all_possible_motifs = list(combinations_with_replacement(univariate_motifs, dimension))
    sets = _generate_unique_sets(len(univariate_motifs), dimension, nb_motif_sets, rng)
    motifs = [[univariate_motifs[i] for i in s] for s in sets]
    # motifs = list(map(list, rng.choice(all_possible_motifs, nb_motif_sets, replace=False)))
    for motif, (mask, location) in zip(motifs, motif_sets):
        i = -1
        for d in range(dimension):
            if not mask[d]:
                continue
            i += 1
            for k in range(location.shape[0]):
                start = location[k, 0, i]
                end = location[k, 1, i]
                X[start:end, d] = motif[d](end - start)

    # Add noise to the time series
    X[np.where(available_positions)] = rng.normal(
        loc=0, scale=noise_non_motif, size=X.shape
    )[np.where(available_positions)]
    X += rng.normal(scale=noise_general, size=X.shape)

    # Return the data
    return X, motif_sets


def generate_tsmd_benchmark_ts(df, dimension, g=2):
    freqs = df["label"].value_counts()

    classes = freqs.index
    classes = classes[freqs > 1]

    if len(classes) < g:
        ValueError("TODO")

    # Pick the repeating classes randomly
    repeating_classes = np.random.choice(classes, size=g, replace=False)
    # Sample one instance of every non-repeating class. Then randomly order them
    X_non_repeating = df[~df["label"].isin(repeating_classes)].copy()
    X_non_repeating = X_non_repeating.groupby("label", group_keys=False).apply(
        lambda x: x.sample()
    )
    X_non_repeating = X_non_repeating.sample(frac=1).reset_index(drop=True)

    # Sample at least two motifs for each repeating class
    X_repeating = df[df["label"].isin(repeating_classes)].reset_index(drop=True)
    motifs = X_repeating.groupby("label", group_keys=False).apply(
        lambda x: x.sample(n=2)
    )

    # Then complete the GT motifs by randomly sampling other motifs from repeating classes
    other_motifs = X_repeating[
        ~X_repeating.apply(tuple, 1).isin(motifs.apply(tuple, 1))
    ]
    other_motifs = other_motifs.sample(
        n=min(len(other_motifs), max(0, len(X_non_repeating) + 1 - len(motifs))),
        replace=False,
    )

    all_motifs = pd.concat((motifs, other_motifs))
    all_motifs = all_motifs.sample(frac=1).reset_index(drop=True)

    gt = {c: [] for c in repeating_classes}
    ts = []

    dimensions_in_data = motifs.iloc[0, 0].shape[1]
    selected_dimension = {
        c: cols
        for c, cols in zip(
            repeating_classes, _select_values(dimension, dimensions_in_data, g)
        )
    }

    # Concatenate instances, alternating between a non-repeating and a repeating class, until no instances left
    curr = 0
    for i in range(min(len(X_non_repeating), len(all_motifs))):

        # Non-motif
        selected_columns = np.random.choice(
            dimensions_in_data, dimension - dimensions_in_data, replace=True
        )
        all_columns = np.concatenate((np.arange(dimensions_in_data), selected_columns))
        np.random.shuffle(all_columns)
        instance, _, l = X_non_repeating.iloc[i]
        instance = instance[:, all_columns]
        ts.append(instance)
        curr += l

        # Motif
        motif, label, l = all_motifs.iloc[i]
        higher_dimensional_motif = np.zeros(shape=(l, dimension))
        higher_dimensional_motif[:, selected_dimension[label]] = motif
        for d in range(dimension):
            if d in selected_dimension[label]:
                continue
            new_dimension = np.random.choice(list(set(range(dimension)) - {d}))
            higher_dimensional_motif[:, d] = scipy.signal.resample(
                instance[:, new_dimension], l
            )
        ts.append(higher_dimensional_motif)
        gt[label].append((curr, curr + l))
        curr += l

    # Format the ground truth
    gt_formatted = []
    for k, v in gt.items():
        mask = np.zeros(shape=dimension, dtype=bool)
        mask[selected_dimension[k]] = True
        motifs = np.repeat(np.array(np.array(v)), dimensions_in_data).reshape(
            len(v), 2, dimensions_in_data
        )
        motifs -= ts[0].shape[0]  # Because this one will be removed
        gt_formatted.append((mask, motifs))

    # Format the ground truth
    return np.vstack(ts[1:]), gt_formatted


def generate_tsmd_benchmark_dataset(df, N, g_min, g_max):
    # Generate time series
    benchmark_ts = []
    gts = []
    for _ in range(N):
        # Sample a number of motif sets
        g = np.random.randint(g_min, g_max + 1)

        dimension_data = df.iloc[0, 0].shape[1]
        if g == 1:
            dimension = dimension_data + 1
        else:
            dimension = np.random.randint(dimension_data + 1, g * dimension_data + 1)
        ts, gt = generate_tsmd_benchmark_ts(df, dimension=dimension, g=g)

        benchmark_ts.append(ts)
        gts.append(gt)

    benchmark_dataset = pd.DataFrame({"ts": benchmark_ts, "gt": gts})
    return benchmark_dataset
