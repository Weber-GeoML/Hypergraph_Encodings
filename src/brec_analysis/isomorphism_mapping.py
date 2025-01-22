"""Functions for finding the isomorphism mapping between two graphs.

Note: pairs in BREC are not isomorphic.
"""


import networkx as nx
import networkx.algorithms.isomorphism as iso



def find_isomorphism_mapping(G1, G2):
    """
    Find the node mapping between two isomorphic graphs with detailed debugging.
    """

    # Convert graphs to simple undirected graphs
    G1 = nx.Graph(G1)
    G2 = nx.Graph(G2)

    print("\n=== Detailed Isomorphism Check ===")
    print("\nGraph Properties:")
    print(f"G1: {len(G1)} nodes, {G1.number_of_edges()} edges")
    print(f"G2: {len(G2)} nodes, {G2.number_of_edges()} edges")

    print("\nNode Degrees:")
    print("G1 degrees:", sorted([d for n, d in G1.degree()]))
    print("G2 degrees:", sorted([d for n, d in G2.degree()]))

    print("\nEdge Lists:")
    print("G1 edges:", sorted(G1.edges()))
    print("G2 edges:", sorted(G2.edges()))

    class VerboseGraphMatcher(iso.GraphMatcher):
        def __init__(self, G1, G2):
            super().__init__(G1, G2)
            self.mapping_steps = []

        def semantic_feasibility(self, G1_node, G2_node):
            """Print detailed information about node matching attempts"""
            feasible = super().semantic_feasibility(G1_node, G2_node)
            print("\nTrying to match:")
            print(f"G1 node {G1_node} (degree {G1.degree[G1_node]}) with")
            print(f"G2 node {G2_node} (degree {G2.degree[G2_node]})")
            print(f"Current mapping: {self.mapping}")
            print(f"Feasible: {feasible}")
            if not feasible:
                print("Neighbors comparison:")
                print(f"G1 node {G1_node} neighbors: {list(G1.neighbors(G1_node))}")
                print(f"G2 node {G2_node} neighbors: {list(G2.neighbors(G2_node))}")
            return feasible

    try:
        # Create verbose graph matcher
        GM = VerboseGraphMatcher(G1, G2)

        # Check isomorphism
        is_isomorphic = GM.is_isomorphic()

        if is_isomorphic:
            mapping = GM.mapping
            print("\n✅ Graphs are isomorphic!")
            print(f"Final mapping: {mapping}")

            # Verify mapping
            print("\nVerifying mapping...")
            for edge in G1.edges():
                mapped_edge = (mapping[edge[0]], mapping[edge[1]])
                if not G2.has_edge(*mapped_edge):
                    print(f"❌ Mapping verification failed for edge {edge}!")
                    return None
                print(f"✓ Edge {edge} correctly maps to {mapped_edge}")

            return mapping
        else:
            print("\n❌ Graphs are not isomorphic")
            print("Possible reasons:")
            print("1. Different degree sequences")
            print("2. Different neighborhood structures")
            print("3. No valid node mapping preserves all edges")
            return None

    except Exception as e:
        print(f"\n❌ Error during isomorphism check: {str(e)}")
        return None
