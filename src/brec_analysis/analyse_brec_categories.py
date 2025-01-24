import numpy as np
import os
import networkx as nx

def analyze_brec_categories() -> dict:
    """Analyse the BREC dataset by category"""

    # Define categories and their files
    categories : dict = {
        "basic": "basic.npy",
        "regular": "regular.npy",
        "str": "str.npy",  # strongly regular
        "cfi": "cfi.npy",
        "extension": "extension.npy",
        "4vtx": "4vtx.npy",
        "dr": "dr.npy"  # distance regular
    }

    data_path = "BREC_Data"
    print(f"\nLoading data from: {data_path}")
    print("\nBREC Dataset Structure:")
    total_pairs = 0
    total_graphs = 0
    
    # Dictionary to store graphs by category
    graphs_by_category = {}
    
    for category, filename in categories.items():
        file_path = os.path.join(data_path, filename)
        try:
            data = np.load(file_path, allow_pickle=True)
            num_pairs = len(data) // 2
            total_pairs += num_pairs
            total_graphs += len(data)
            print(f"{category}: {num_pairs} pairs ({len(data)} graphs)")
            
            # Convert to NetworkX graphs and store
            nx_graphs = []
            for graph_data in data:
                G = nx.Graph()
                nodes = range(len(graph_data))  # graph_data[0] is the number of nodes
                edges = graph_data[1]  # graph_data[1] contains the edges
                G.add_nodes_from(nodes)
                G.add_edges_from(edges)
                nx_graphs.append(G)
            
            graphs_by_category[category] = nx_graphs
            
            # Print info about first graph
            first_graph = nx_graphs[0]
            print(f"  First graph: {first_graph.number_of_nodes()} nodes, {first_graph.number_of_edges()} edges")
            
        except Exception as e:
            print(f"Error loading {category}: {e}")

    print(f"\nTotal: {total_pairs} pairs ({total_graphs} graphs)")
    
    return graphs_by_category

