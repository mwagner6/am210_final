import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from gaussianity_test import test_gaussianity
from gds_core import gds
from lingam import run_lingam
from grasp import run_grasp

# Load csv data (named by first entry)
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    variable_names = df.columns.tolist()
    X = df.values
    return X, variable_names

# Create and save causal graph
def visualize_graph(adj, variable_names, method, output_path):
    p = len(variable_names)

    G = nx.DiGraph()
    G.add_nodes_from(range(p))

    # Add directed edges
    for i in range(p):
        for j in range(p):
            if adj[i, j] == 1:
                G.add_edge(i, j)

    # Record undirected edges to add
    undirected_edges = []
    for i in range(p):
        for j in range(i + 1, p):
            if adj[i, j] == 2 and adj[j, i] == 2:
                undirected_edges.append((i, j))

    fig, ax = plt.subplots(figsize=(12, 10))

    pos = nx.spring_layout(G, k=2, iterations=50, seed=210)

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000, ax=ax)

    # Draw directed edges
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True, arrowsize=50,
                            arrowstyle='->', width=2.5, ax=ax)

    # Draw undirected edges
    if undirected_edges:
        nx.draw_networkx_edges(G, pos, edgelist=undirected_edges, edge_color='red', 
                                width=2.5, arrows=False, ax=ax)

    labels = {i: variable_names[i] for i in range(p)}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold', ax=ax)

    graph_type = "Partial DAG (Markov Equivalence Class)" if method == "GrASP" else "DAG"
    ax.set_title(f"Causal Graph - {method}\n({graph_type})",
                fontsize=16, fontweight='bold')

    legend_elements = [
        plt.Line2D([0], [0], color='black', linewidth=2.5,
                    label='Directed edge', marker='>', markersize=10)
    ]
    if undirected_edges:
        legend_elements.append(
            plt.Line2D([0], [0], color='red', linewidth=2.5,
                        linestyle='dashed', label='Undirected edge')
        )
    ax.legend(handles=legend_elements, loc='upper right')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_outputs(result, variable_names, method, base_path):
    # Get adjacency matrix
    if 'cpdag' in result:
        adj = result['cpdag']
    else:
        adj = result['Adj']

    # We can't infer edge weights or error variances from MEC
    base_path = os.path.join("results/", base_path)
    os.makedirs(base_path, exist_ok=True)
    if method != "GrASP":
        B = result['B']
        variances = result['variances']

        edge_weights_df = pd.DataFrame(B, index=variable_names, columns=variable_names)
        edge_weights_path = f"{base_path}/edge_weights.csv"
        edge_weights_df.to_csv(edge_weights_path)

        variances_df = pd.DataFrame({'Variable': variable_names, 'Variance': variances})
        variances_path = f"{base_path}/variances.csv"
        variances_df.to_csv(variances_path, index=False)

    graph_path = f"{base_path}/graph.png"
    visualize_graph(adj, variable_names, method, graph_path)


def run_pipeline(csv_path, similar_variance):
    X, variable_names = load_data(csv_path)
    n, p = X.shape

    gauss_results = test_gaussianity(X)

    if gauss_results == 0:
        print("All non-Gaussian: running LiNGAM")
        result = run_lingam(X)
        method = "LiNGAM"
    elif gauss_results == 1 and similar_variance:
        print("All Gaussian and similar variance: running GDS")
        result = gds(X)
        method = "GDS"
    else:
        print("Mixed Gaussian or unknown error variance: Running GRaSP")
        result = run_grasp(X)
        method = "GrASP"

    save_outputs(result, variable_names, method, Path(csv_path).stem)

def main():
    if len(sys.argv) != 3:
        sys.exit(1)

    csv_path = sys.argv[1]
    similar_variance_str = sys.argv[2]

    if not os.path.exists(csv_path):
        print("Invalid file path")
        sys.exit(1)

    if similar_variance_str.lower() == "true":
        similar_variance = True
    elif similar_variance_str.lower() == "false":
        similar_variance = False
    else:
        print("Similar variance input must be either 'true' or 'false'.")
        sys.exit(1)

    run_pipeline(csv_path, similar_variance)


if __name__ == '__main__':
    main()
