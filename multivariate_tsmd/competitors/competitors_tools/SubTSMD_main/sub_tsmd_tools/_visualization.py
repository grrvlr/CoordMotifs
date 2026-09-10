import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D

from ._SubspaceMotifSet import SubspaceMotifSet


def _format_motif_sets(
    y: list[SubspaceMotifSet] | list[(np.array, np.ndarray)]
) -> list[SubspaceMotifSet]:
    return [
        (
            motif_set
            if isinstance(motif_set, SubspaceMotifSet)
            else SubspaceMotifSet.from_arrays(motif_set)
        )
        for motif_set in y
    ]


def plot_motif_sets(
    X: np.ndarray,
    y: list[SubspaceMotifSet] | list[(np.array, np.ndarray)],
    time_steps: np.array = None,
    figsize: (float, float) = (20, 7),
    alpha_patches: float = 0.2,
    color_cycle: list[str] = None,
) -> plt.Figure:
    # Format the motif sets
    y = _format_motif_sets(y)

    # Set up the figure
    fig, axs = plt.subplots(X.shape[1], figsize=figsize, sharex="all")
    axs = [axs] if X.shape[1] == 1 else axs

    # Initialize the color cycle
    if color_cycle is None:
        color_cycle = [cm.jet(x) for x in np.linspace(0.0, 1.0, len(y))]

    # Format the time steps
    if time_steps is None:
        time_steps = np.arange(X.shape[0], dtype=float)
    # Plot the time series
    for d in range(X.shape[1]):
        axs[d].plot(time_steps, X[:, d], color="gray")

    # Mark the motifs
    for i, motif_set in enumerate(y):
        for motif in motif_set:
            for attribute in motif.subspace:
                axs[attribute].axvspan(
                    time_steps[int(motif.start(attribute))],
                    time_steps[min(int(motif.end(attribute)), len(time_steps) - 1)],
                    color=color_cycle[i],
                    alpha=alpha_patches,
                )

    return fig


def plot_motif_sets_marking(
    X: np.ndarray,
    y: list[SubspaceMotifSet] | list[(np.array, np.ndarray)],
    time_steps: np.array = None,
    figsize: (float, float) = (20, 7),
    linewidth: float = 10.0,
    alpha_marking: float = 0.5,
    color_cycle: list[str] = None,
) -> plt.Figure:
    # Format the motif sets
    y = _format_motif_sets(y)

    # Set up the figure
    fig, axs = plt.subplots(X.shape[1], figsize=figsize, sharex="all")
    axs = [axs] if X.shape[1] == 1 else axs

    # Initialize the color cycle
    if color_cycle is None:
        color_cycle = [cm.jet(x) for x in np.linspace(0.0, 1.0, len(y))]

    # Format the time steps
    if time_steps is None:
        time_steps = np.arange(X.shape[0], dtype=float)

    # Plot the time series
    for d in range(X.shape[1]):
        axs[d].plot(time_steps, X[:, d], color="gray")

    # Mark the motifs
    for i, motif_set in enumerate(y):
        for motif in motif_set:
            for attribute in motif.subspace:
                start_index = np.argmin(np.abs(time_steps - motif.start(attribute)))
                end_index = np.argmin(np.abs(time_steps - motif.end(attribute)))
                axs[attribute].plot(
                    time_steps[start_index:end_index],
                    X[start_index:end_index, attribute],
                    color=color_cycle[i],
                    alpha=alpha_marking,
                    linewidth=linewidth,
                )

    return fig


def plot_motif_sets_independent(
    X: np.ndarray,
    y: list[SubspaceMotifSet] | list[(np.array, np.ndarray)],
    height_ratio: int = 0.5,
) -> plt.Figure:
    # Format the motif sets
    y = _format_motif_sets(y)

    # Set up the figure
    height = sum([motif_set.mask.sum() for motif_set in y]) * height_ratio
    fig, axs = plt.subplots(
        nrows=len(y),
        figsize=(20, height),
        sharex="all",
        height_ratios=[motif_set.mask.sum() for motif_set in y],
    )
    if len(y) == 1:
        axs = [axs]

    # Plot the motifs
    color_cycle = [cm.jet(x) for x in np.linspace(0.0, 1.0, X.shape[1])]

    combined_mask = np.zeros_like(y[0].mask)
    for motif_set, ax in zip(y, axs):
        combined_mask |= motif_set.mask
        for i, attribute in enumerate(motif_set[0].subspace):
            data = (X[:, attribute] - X[:, attribute].min()) / (
                X[:, attribute].max() - X[:, attribute].min()
            )
            ax.plot(data - i * 1.02, c=color_cycle[attribute])
            ax.set_yticks([])

            for motif in motif_set:
                ax.add_patch(
                    plt.Rectangle(
                        (motif.start(attribute), -i * 1.02),
                        motif.end(attribute) - motif.start(attribute),
                        1,
                        color=color_cycle[attribute],
                        alpha=0.2,
                    )
                )

    # Final formatting
    legend_elements = [
        Line2D([0], [0], color=color_cycle[i], label=f"Dimension {i}")
        for i, attribute_used in enumerate(combined_mask)
        if attribute_used
    ]
    axs[0].legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        ncols=len(legend_elements),
    )

    return fig
