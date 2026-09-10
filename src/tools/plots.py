import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np

def get_scc_roles(G):
    """
    Compute SCCs and assign a role (source / sink / intermediate / isolated)
    to each SCC, then map this role to original nodes.

    Returns:
        node_role: dict {node: role_str}
        sccs: list of sets
        C: condensed DAG
    """
    # SCC decomposition
    sccs = list(nx.strongly_connected_components(G))

    # Condensed graph (DAG)
    C = nx.condensation(G, sccs)

    # Determine SCC roles
    scc_role = {}
    for scc_id in C.nodes:
        indeg = C.in_degree(scc_id)
        outdeg = C.out_degree(scc_id)

        if indeg == 0 and outdeg == 0:
            role = "isolated"
        elif indeg == 0:
            role = "source"
        elif outdeg == 0:
            role = "sink"
        else:
            role = "intermediate"

        scc_role[scc_id] = role

    # Map node -> role
    node_role = {}
    for scc_id, scc in enumerate(sccs):
        for node in scc:
            node_role[node] = scc_role[scc_id]

    return node_role, sccs, C

def plot_directed_cc_graph_scc_roles(
    G,
    node_size=900,
    edge_width_scale=5
):
    """
    Plot directed CC graph, coloring nodes by SCC role (source / sink / etc.).
    """
    node_role, sccs, C = get_scc_roles(G)

    # Color map by role
    role_color = {
        "source": "limegreen",
        "sink": "tomato",
        "intermediate": "lightgray",
        "isolated": "gold",
    }

    node_colors = [role_color[node_role[n]] for n in G.nodes]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    # Nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_colors,
        edgecolors="k",
        linewidths=1.0
    )

    # Edges
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=15,
        width=[w * edge_width_scale for w in weights],
        edge_color=weights,
        edge_cmap=plt.cm.viridis
    )

    # Edge labels
    edge_labels = {
        (u, v): f"{G[u][v]['weight']:.2f}"
        for u, v in G.edges()
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

    # Node labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Legend
    
    legend_handles = [
        mpatches.Patch(color="limegreen", label="SCC source (motif atomique)"),
        mpatches.Patch(color="tomato", label="SCC puits (fusion ambiguë)"),
        mpatches.Patch(color="lightgray", label="SCC intermédiaire"),
        mpatches.Patch(color="gold", label="SCC isolée"),
    ]
    plt.legend(handles=legend_handles, loc="best")

    plt.title("Directed CC graph — SCC source / sink highlighting")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def get_sink_and_ancestors_sccs(G, coverage_scores):
    sink_ids = [n for n in G.nodes if G.out_degree(n) == 0]
    sinks_with_ancestors = []

    ancestors = []
    for sink_id in sink_ids:
        anc_ids = nx.ancestors(G, sink_id)
        if not anc_ids:
            continue
        anc_ids_sorted = sorted(
            anc_ids,
            key=lambda n: coverage_scores.get(n, 0),
            reverse=True
        )
        ancestors.append(anc_ids_sorted)
        sinks_with_ancestors.append(sink_id)

    # trier les sinks par coverage décroissant
    sinks_with_ancestors = sorted(
        sinks_with_ancestors,
        key=lambda n: coverage_scores.get(n, 0),
        reverse=True
    )
    return sinks_with_ancestors, ancestors


def plot_candidates(candidates,n_samples,n_dims):
    cmap = cm.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(18, 6))

    for i, (_, (dim_list, pos_list)) in enumerate(candidates.items()):
        if len(pos_list) == 0:
            continue
        points = np.array([(d, p) for d in dim_list for p in pos_list])
        ax.scatter(points[:, 1], points[:, 0], s=1, color=cmap(i % cmap.N))

    ax.set_xlim(0, n_samples)
    ax.set_ylim(n_dims - 0.5, -0.5)
    ax.set_xlabel("Position")
    ax.set_ylabel("Dimension")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
