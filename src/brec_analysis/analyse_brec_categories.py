import numpy as np
import os
import networkx as nx

def analyze_brec_categories() -> dict:
    """Analyse the BREC dataset by category
    
    Returns:
        dict: Dictionary mapping categories to lists of NetworkX graphs
    """
    categories: dict = {
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
    
    graphs_by_category: dict = {}
    
    for category, filename in categories.items():
        file_path = os.path.join(data_path, filename)
        try:
            data = np.load(file_path, allow_pickle=True)
            num_pairs = len(data) // 2
            total_pairs += num_pairs
            total_graphs += len(data)
            print(f"{category}: {num_pairs} pairs ({len(data)} graphs)")
            
            nx_graphs = []
            if category in ["regular", "cfi"]:
                # Handle array of pairs format
                for pair in data:
                    for g6_bytes in pair:
                        G = nx.from_graph6_bytes(g6_bytes)
                        nx_graphs.append(G)
            elif category == "extension":
                # Handle extension format
                for pair in data:
                    for g6_str in pair:
                        G = nx.from_graph6_bytes(g6_str.encode())
                        nx_graphs.append(G)
            else:
                # Handle basic format (alternating graphs)
                for g6_str in data:
                    if isinstance(g6_str, bytes):
                        G = nx.from_graph6_bytes(g6_str)
                    else:
                        G = nx.from_graph6_bytes(g6_str.encode())
                    nx_graphs.append(G)
            
            graphs_by_category[category] = nx_graphs
            
            # Print info about first graph
            first_graph = nx_graphs[0]
            print(f"  First graph: {first_graph.number_of_nodes()} nodes, "
                  f"{first_graph.number_of_edges()} edges")
            
        except Exception as e:
            print(f"Error loading {category}: {e}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
                print(f"First item content: {data[0]}")

    print(f"\nTotal: {total_pairs} pairs ({total_graphs} graphs)")
    
    return graphs_by_category

