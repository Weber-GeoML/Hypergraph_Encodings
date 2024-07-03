"""
curvatures_orc.py

This module contains functions for computing the Ollivier-Ricci curvatures of an undirected hypergraph.
"""

import json
import os
import subprocess
import sys

from encodings_hnns.data_handling import parser

print(sys.path)

# from julia import Julia
# julia = Julia()

import subprocess


def map_nodes_to_integers(hypergraph):
    """
    Reason I am doing this is because Julia code
    assumes that the nodes are [1,2,3,...,n]
    where n is the number of nodes.

    Example:
    in:
    hypergraph = {"y": [5, 6], "g": [8, 10]}
    out:
    {'y': [1, 2], 'g': [3, 4]}
    {5: 1, 6: 2, 8: 3, 10: 4}
    {1: 5, 2: 6, 3: 8, 4: 10}
    """
    node_to_int = {}
    int_to_node = {}
    next_int = 1  # Start assigning integers from 1

    # Extract all unique nodes from the hypergraph
    all_nodes = set()
    for nodes_list in hypergraph.values():
        all_nodes.update(nodes_list)

    # Assign integer IDs to nodes
    for node in sorted(all_nodes):
        node_to_int[node] = next_int
        int_to_node[next_int] = node
        next_int += 1

    # Map nodes in the hypergraph to their integer IDs
    mapped_hypergraph = {}
    for hedge, nodes in hypergraph.items():
        mapped_nodes = [node_to_int[node] for node in nodes]
        mapped_hypergraph[hedge] = mapped_nodes

    return mapped_hypergraph, node_to_int, int_to_node


def _save_to_tsv(hypergraph: dict, output_file: str) -> None:
    """Saves hypergraph edges to a TSV file with exactly three spaces between nodes.

    Then we pass this in to the julia code.

    Args:
        hypergraph (dict): The hypergraph dictionary.
        output_file (str): The path to the output TSV file.
    """
    # Open the file in write mode
    with open(output_file, "w") as f:
        # Iterate over each hyperedge (key) and its associated nodes (values)
        for hedge, nodes in hypergraph.items():
            # Write the nodes to the file as a single line with three spaces between nodes
            line = "   ".join(map(str, nodes)) + "\n"
            f.write(line)


class ORC:
    def __init__(self, hypergraph: dict) -> None:
        """
        Initialize the Forman-Ricci curvature object.
        A hypergraph is a dictionary with the following keys:
        - hypergraph : a dictionary with the hyperedges as values
        - features : a dictionary with the features of the nodes as values
        - labels : a dictionary with the labels of the nodes as values
        - n : the number of nodes in the hypergraph
        """
        assert "hypergraph" in hypergraph.keys(), "Hypergraph not found."
        assert "features" in hypergraph.keys(), "Features not found."
        assert "labels" in hypergraph.keys(), "Labels not found."
        assert "n" in hypergraph.keys(), "Number of nodes not found."

        self.hypergraph: dict = hypergraph
        self.node_curvature_edges = None
        self.node_curvature_neighborhood = None
        self.edge_curvature = None

    def compute_orc(
        self,
        dispersion: str = "UnweightedClique",
        alpha: float = 0,
        Aggregation_type: str = "Mean",
    ) -> None:
        """Computes the ORC for hypergraphs.

        We get the neighbourhood agregation, edge aggregation.
        We also get edges ORC.

        Args:
            Dispersions:
                One of "UnweightedClique", "UnweightedStar", "WeightedClique
            alpha:
            Aggregation_type:
                Mean or Max. See eq 8, 10 (not implement) and 11 in Choupette (Orchid)
                Mean:
                This is equivalent to computing the curvature of e based on the average over all W1 distances of
                probability measures associated with nodes contained in
                Max:
                captures the maximum amount
                of work needed to transport all probability mass from one node in e to another node in

        """

        hypergraph: dict = self.hypergraph["hypergraph"]

        hypergraph = {"y": [5, 6], "g": [8, 10]}

        hypergraph, b, c = map_nodes_to_integers(hypergraph=hypergraph)

        # Get the directory of the current script
        script_dir = os.path.dirname(__file__)

        # Define paths relative to the current script
        orchid_jl_path = os.path.abspath(
            os.path.join(script_dir, "../../src/orchid/orchid_interface.jl")
        )
        input_file = os.path.abspath(os.path.join(script_dir, "hypergraph_edges.tsv"))
        result_file = os.path.abspath(
            os.path.join(
                script_dir, f"results.alpha-{alpha}.dispersion-{dispersion}.orc.json"
            )
        )

        # Specify the output file path
        _save_to_tsv(hypergraph=hypergraph, output_file=input_file)

        # Get the current working directory
        current_path = os.getcwd()

        # Print the current working directory
        print("Current working directory:", current_path)

        # Option 1:
        # Transform the hypergraph into the Format needed for julia

        # Option 2: (at a later stage)
        # transform the julia code to read in the hypergraph

        # Subroutine to call Julia code

        # Define the command to execute
        command = [
            "julia",
            orchid_jl_path,
            "--aggregation",
            "All",
            "--dispersion",
            str(dispersion),
            "--alpha",
            str(alpha),
            "-i",
            input_file,
            "-o",
            result_file,
        ]

        # Execute the command using subprocess
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")

        # the we read
        data: list[dict]
        with open(result_file, "r") as f:
            data = json.load(f)

        # at this stage data is a list of two dicts
        # first one uses mean aggregation
        # second ones uses max aggregation
        stats: dict
        if Aggregation_type == "Mean":
            stats = data[0]
        elif Aggregation_type == "Max":
            stats = data[1]

        print(stats)

        self.node_curvature_edges = stats["node_curvature_edges"]
        self.node_curvature_neighborhood = stats["node_curvature_neighborhood"]
        self.edge_curvature = stats["edge_curvature"]


# Example utilization
if __name__ == "__main__":

    data_type = "coauthorship"
    dataset_name = "cora"

    # Create an instance of the parser class
    parser_instance = parser(data_type, dataset_name)

    data = parser_instance._load_data()
    # print(data["hypergraph"])
    # So hypergraph is a dict:
    # key: authors, values: papers participates in.
    print(data["features"])
    print(data["labels"])

    # Instantiates the ORC class
    orc = ORC(data)

    # Computes the Forman-Ricci curvature
    orc.compute_orc()

    # Accesses the results
    print("ORC:", orc.node_curvature_edges)
