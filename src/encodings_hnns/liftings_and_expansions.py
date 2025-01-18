"""Code for lifting (graph to hypergraph) and expanding (hypergraph to graph)
Run:
pip install --no-deps hypernetx
pip install fastjsonschema


"""
import networkx as nx
import itertools
import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import hypernetx as hnx

def lift_to_hypergraph(graph) -> dict:
    """
    Constructs a hypergraph from a given graph by identifying maximal cliques of size >=3
    and including remaining edges as hyperedges of size 2.
    
    Parameters:
    - graph (torch_geometric.data.Data): The input graph with attributes x (features) and y (labels).
    
    Returns:
    - dataset (dict): A dictionary containing the hypergraph, number of nodes, features, and labels.
        - 'hypergraph': dict mapping hyperedge IDs to lists of node indices.
        - 'n': int, number of nodes.
        - 'features': list of lists, node feature vectors.
        - 'labels': list, node labels.

    Note: function given by Lukas.
    """
    # Convert PyG graph to NetworkX format for clique detection
    G = to_networkx(graph, to_undirected=True)
    
    # Find all maximal cliques of size 3 or larger
    cliques = [clique for clique in nx.find_cliques(G) if len(clique) >= 3]
    print(f"Found {len(cliques)} cliques")
    print(f"The cliques are {cliques}")
    
    # Keep track of edges that are part of cliques (to avoid duplicates)
    edges_in_cliques = set()
    for clique in cliques:
        # For each clique, get all possible pairs of nodes (edges)
        for edge in itertools.combinations(clique, 2):
            edges_in_cliques.add(tuple(sorted(edge)))  # Store edges in sorted order for consistency
            print(f"The edge {edge} is in the clique {clique}")
    
    # Get all edges from the graph, ensuring consistent ordering
    all_edges = set(tuple(sorted(edge)) for edge in G.edges())
    print(f"The all edges are {all_edges}")
    # Find edges that aren't part of any clique
    remaining_edges = all_edges - edges_in_cliques
    print(f"The remaining edges are {remaining_edges}")
    # Create the hypergraph dictionary
    hypergraph : dict = {}
    hyperedge_id : int = 0
    
    # Add cliques as hyperedges (size >= 3)
    for clique in cliques:
        hypergraph[f'he_{hyperedge_id}'] = list(clique)
        hyperedge_id += 1
    
    # Add remaining edges as size-2 hyperedges
    for edge in remaining_edges:
        hypergraph[f'he_{hyperedge_id}'] = list(edge)
        hyperedge_id += 1
    
    # Get number of nodes
    n = graph.num_nodes
    
    # Extract node features if they exist, otherwise create empty feature lists
    features = graph.x.cpu().numpy().tolist() if hasattr(graph, 'x') else [[] for _ in range(n)]
    # Extract node labels if they exist, otherwise create default labels
    labels = graph.y.cpu().tolist() if hasattr(graph, 'y') else [0 for _ in range(n)]
    
    # Create final dataset dictionary with all components
    dataset : dict = {
        'hypergraph': hypergraph,
        'n': n,
        'features': features,
        'labels': labels
    }
    return dataset

def example_lifting():
    """Example of lifting a simple graph to a hypergraph"""
    # Create a simple graph with 4 nodes
    # Let's make a graph with a triangle (clique of size 3) and one additional edge
    edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 3, 0, 5, 5, 6],  # Source nodes
                              [1, 2, 3, 2, 3, 3, 2, 5, 6, 7, 7]], dtype=torch.long)  # Target nodes
    
    # Add dummy node features (1-dimensional for simplicity)
    x = torch.ones((4, 1), dtype=torch.float)  # 4 nodes, 1 feature each
    
    # Add dummy labels (one label per node)
    y = torch.zeros(4, dtype=torch.long)  # 4 nodes, all labeled as class 0
    
    # Create PyG Data object with features and labels
    graph = Data(x=x, y=y, edge_index=edge_index, num_nodes=4)
    
    # Apply lifting
    hypergraph_dict = lift_to_hypergraph(graph)
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot original graph
    plt.subplot(121)
    G = to_networkx(graph, to_undirected=True)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, font_weight='bold')
    plt.title("Original Graph")
    
    # Plot hypergraph
    plt.subplot(122)
    
    # Create HyperNetX hypergraph
    H = hnx.Hypergraph(hypergraph_dict['hypergraph'])
    
    # Draw the hypergraph with correct parameters
    hnx.draw(H, 
            with_node_labels=True,
            with_edge_labels=True)
            # nodes_kwargs={'color': 'lightblue'},
            # edges_kwargs={'color': 'red', 'alpha': 0.2})
    
    plt.title("Hypergraph Representation")
    
    plt.tight_layout()
    plt.show()

def lift_and_plot_graphs():
    """Lift and plot Rook and Shrikhande graphs to hypergraphs"""
    
    # Load the graphs from g6 format
    shrikhande = nx.read_graph6("shrikhande.g6")
    rooke = nx.read_graph6("rook_graph.g6")
    
    # Convert NetworkX graphs to PyG Data objects
    def nx_to_pyg(G):
        # Get edge index
        edge_index = torch.tensor([[e[0] for e in G.edges()], 
                                 [e[1] for e in G.edges()]], dtype=torch.long)
        # Add dummy features and labels
        x = torch.ones((G.number_of_nodes(), 1), dtype=torch.float)
        y = torch.zeros(G.number_of_nodes(), dtype=torch.long)
        return Data(x=x, y=y, edge_index=edge_index, num_nodes=G.number_of_nodes())
    
    shrikhande_pyg = nx_to_pyg(shrikhande)
    rooke_pyg = nx_to_pyg(rooke)
    
    # Lift both graphs to hypergraphs
    shrikhande_hyper = lift_to_hypergraph(shrikhande_pyg)
    rooke_hyper = lift_to_hypergraph(rooke_pyg)
    
    # Plot Shrikhande graph and its hypergraph
    plt.figure(figsize=(15, 7))
    plt.suptitle("Shrikhande Graph", fontsize=16)
    
    # Original graph
    plt.subplot(121)
    pos = nx.circular_layout(shrikhande)
    nx.draw(shrikhande, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=12, font_weight='bold')
    plt.title("Original Graph")
    
    # Shrikhande hypergraph
    plt.subplot(122)
    H_shrikhande = hnx.Hypergraph(shrikhande_hyper['hypergraph'])
    hnx.draw(H_shrikhande, with_node_labels=True, with_edge_labels=False)
    plt.title(f"Hypergraph Representation\n({len(shrikhande_hyper['hypergraph'])} hyperedges)")
    
    plt.tight_layout()
    plt.savefig('shrikhande_lifting.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot Rooke graph and its hypergraph
    plt.figure(figsize=(15, 7))
    plt.suptitle("Rooke Graph", fontsize=16)
    
    # Original graph
    plt.subplot(121)
    pos = nx.circular_layout(rooke)
    nx.draw(rooke, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=12, font_weight='bold')
    plt.title("Original Graph")
    
    # Rooke hypergraph
    plt.subplot(122)
    H_rooke = hnx.Hypergraph(rooke_hyper['hypergraph'])
    hnx.draw(H_rooke, with_node_labels=True, with_edge_labels=False)
    plt.title(f"Hypergraph Representation\n({len(rooke_hyper['hypergraph'])} hyperedges)")
    
    plt.tight_layout()
    plt.savefig('rooke_lifting.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print some statistics
    print("\nShrikhande Graph Statistics:")
    print(f"Number of nodes: {shrikhande.number_of_nodes()}")
    print(f"Number of edges: {shrikhande.number_of_edges()}")
    print(f"Number of hyperedges: {len(shrikhande_hyper['hypergraph'])}")
    print("\nRooke Graph Statistics:")
    print(f"Number of nodes: {rooke.number_of_nodes()}")
    print(f"Number of edges: {rooke.number_of_edges()}")
    print(f"Number of hyperedges: {len(rooke_hyper['hypergraph'])}")

if __name__ == "__main__":
    example_lifting()
    lift_and_plot_graphs()