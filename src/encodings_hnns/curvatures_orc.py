"""
curvatures_orc.py

This module contains functions for computing the Ollivier-Ricci curvatures of an undirected hypergraph.
"""

import json
import os
import subprocess
import sys

from encodings_hnns.data_handling import parser

# from julia import Julia
# julia = Julia()


def _map_nodes_to_integers(hypergraph: dict) -> tuple[dict, dict, dict]:
    """
    Reason I am doing this is because Julia code
    assumes that the nodes are [1,2,3,...,n]
    where n is the number of nodes.

    Args:
        hypergraph:
            the hypergraph (a dictionary with the hyperedges as values)

    Returns:
        mapped_hypergraph:
            the same hypergraph, but the nodes have been mapped to the
            [1,2,3,..,n] where n is as small as possible
        node_to_int:
            mapping from node to integers
        int_to_node:
            mapping from interger to node

    Example:
        in:
            hypergraph = {"y": [5, 6], "g": [8, 10]}
        out:
            {'y': [1, 2], 'g': [3, 4]}
            {5: 1, 6: 2, 8: 3, 10: 4}
            {1: 5, 2: 6, 3: 8, 4: 10}
    """
    node_to_int: dict = {}
    int_to_node: dict = {}
    next_int: int = 1  # Starts assigning integers from 1

    # Extracts all unique nodes from the hypergraph
    all_nodes = set()
    for nodes_list in hypergraph.values():
        all_nodes.update(nodes_list)

    # Assigns integer IDs to nodes
    for node in sorted(all_nodes):
        node_to_int[node] = next_int
        int_to_node[next_int] = node
        next_int += 1

    # Map nodes in the hypergraph to their integer IDs
    mapped_hypergraph: dict = {}
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
    line: str
    with open(output_file, "w") as f:
        # Iterate over each hyperedge (key) and its associated nodes (values)
        for hyperedge, nodes in hypergraph.items():
            # Write the nodes to the file as a single line with three spaces between nodes
            line = "   ".join(map(str, nodes)) + "\n"
            f.write(line)


class ORC:
    def __init__(self, hypergraph: dict) -> None:
        """Initialize the ORC curvature object.

        Args:
            hypergraph:
                a hypergraph
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
        self.node_curvature_edges: None | dict = None
        self.node_curvature_neighborhood: None | dict = None
        self.edge_curvature: None | dict = None

    def compute_orc(
        self,
        dispersion: str = "UnweightedClique",
        alpha: float = 0,
        aggregation_type: str = "Mean",
    ) -> None:
        """Computes the ORC for hypergraphs.

        We get the neighbourhood agregation, edge aggregation.
        We also get edges ORC.

        Args:
            dispersions:
                One of "UnweightedClique", "UnweightedStar", "WeightedClique
            alpha:
                TODO
            aggregation_type:
                Mean or Max. See eq 8, 10 (not implement) and 11 in Choupette (Orchid)
                Mean:
                    This is equivalent to computing the curvature of e based on the average over all W1 distances of
                    probability measures associated with nodes contained in
                Max:
                    captures the maximum amount
                    of work needed to transport all probability mass from one node in e to another node in

        """

        hypergraph: dict = self.hypergraph["hypergraph"]

        hypergraph, _, _ = _map_nodes_to_integers(hypergraph=hypergraph)

        # Gets the directory of the current script
        script_dir: str = os.path.dirname(__file__)

        # Defines paths relative to the current script
        orchid_jl_path: str = os.path.abspath(
            os.path.join(script_dir, "../../src/orchid/orchid_interface.jl")
        )
        input_file: str = os.path.abspath(
            os.path.join(script_dir, "hypergraph_edges.tsv")
        )
        result_file: str = os.path.abspath(
            os.path.join(
                script_dir, f"results.alpha-{alpha}.dispersion-{dispersion}.orc.json"
            )
        )

        # Specifies the output file path
        _save_to_tsv(hypergraph=hypergraph, output_file=input_file)

        # Gets the current working directory
        current_path: str = os.getcwd()

        # Print the current working directory
        print("Current working directory:", current_path)

        # Option 1:
        # Transform the hypergraph into the Format needed for julia

        # Option 2: (at a later stage)
        # transform the julia code to read in the hypergraph

        # Subroutine to call Julia code

        # Define the command to execute
        command: list = [
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
        if aggregation_type == "Mean":
            stats = data[0]
        elif aggregation_type == "Max":
            stats = data[1]

        self.node_curvature_edges = stats["node_curvature_edges"]
        print(f"The node curvatures are \n {stats['node_curvature_neighborhood']}")
        self.node_curvature_neighborhood = stats["node_curvature_neighborhood"]
        print(f"The edge curvatures are \n {stats['edge_curvature']}")
        self.edge_curvature = {
            key: stats["edge_curvature"][i]
            for i, key in enumerate(self.hypergraph["hypergraph"])
        }


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
