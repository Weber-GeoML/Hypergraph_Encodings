"""File used to save the encodings for lukas

Will just save in the same format Lukas provided to me.ie a list of dict
And now the "features" field of every dict will have been updated with the features

"""

import multiprocessing as mp
import os
import pickle
import warnings

# necessary for pickle.load
import torch

from encodings_hnns.save_lukas_encodings_class import encodings_saver
from encodings_hnns.save_lukas_encodings_lrgb import encodings_saver_lrgb

warnings.simplefilter("ignore")


# def compute_clique_expansion(dataset: dict) -> torch_geometric.data.Data:
#     """Computes the clique expansion of a hypergraph.

#     This code converts a hypergraph into a regular graph using the clique expansion method.

#     Args:
#         dataset:
#             the hypergraph
#             A hypergraph is a dictionary with the following keys:
#                 - hypergraph : a dictionary with the hyperedges as values
#                 - features : a dictionary with the features of the nodes as values
#                 - labels : a dictionary with the labels of the nodes as values
#                 - n : the number of nodes in the hypergraph
#     """
#     hypergraph: dict[str, list] = dataset["hypergraph"]
#     num_nodes: int = dataset["n"]
#     G: nx.Graph = nx.Graph()
#     # add nodes (note that the nodes go from 0 to n-1 here...)
#     G.add_nodes_from(range(num_nodes))

#     # get edges
#     hyperedges: list[list] = list(hypergraph.values())
#     for hyperedge in hyperedges:
#         #  For each hyperedge, creates regular edges between all
#         # pairs of nodes in that hyperedge.
#         edges = itertools.combinations(hyperedge, 2)
#         G.add_edges_from(edges)

#     # convert to torch geometric data object
#     graph: torch_geometric.data.Data = from_networkx(G)

#     # get node features
#     graph.x = torch.tensor(dataset["features"]).float()

#     # get node labels
#     # (finds the index where the label is 1 in one-hot encoded format)
#     graph.y = torch.tensor([np.where(node == 1)[0][0] for node in dataset["labels"]])
#     return graph


def _convert_to_hypergraph(dataset: torch.Tensor, verbose: bool = False) -> dict:
    """Converts a single graph to hypergraph format.

    THIS IS SPECIFICALLY FOR THE LRGB DATASETS

    Args:
        dataset: Single graph data

    Returns:
        Dictionary containing hypergraph data
    """
    X = dataset[0]  # node features
    if verbose:
        print(f"X shape: {X.shape}")
    edge_attr = dataset[1]  # edge features
    if verbose:
        print(
            f"edge_attr shape: {edge_attr.shape}"
        )  # I don't think we will do anything with these. These are edge features.
    edge_index = dataset[2]  # connectivity
    if verbose:
        print(f"edge_index shape: {edge_index.shape}")
    y = dataset[3]  # labels
    if verbose:
        print(f"y shape: {y.shape}")

    """
    eg:
    X shape: torch.Size([338, 9])
    edge_attr shape: torch.Size([682, 3])
    edge_index shape: torch.Size([2, 682])
    y shape: torch.Size([1, 11])
        
    X shape: torch.Size([338, 9]) - 338 nodes with 9-dimensional features
    edge_attr shape: torch.Size([682, 3]) - 682 edges with 3-dimensional features
    edge_index shape: torch.Size([2, 682]) - 682 edges
    y shape: torch.Size([1, 11]) - an 11-dimensional label for the whole graph
    """
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == edge_attr.shape[0]

    # Convert to hypergraph format
    # We need a dictionary, where the keys are the hyperedges names and the values are hyperdeges (list of the nodes)
    hypergraph: dict[str | int, list[int]] = {}
    for i in range(
        edge_index.shape[1]
    ):  # edge_index.shape[1] gives us the number of edges
        # Create hyperedge from each edge
        hypergraph[f"e_{i}"] = edge_index[:, i].tolist()
        assert len(hypergraph[f"e_{i}"]) == 2

    """
    A hypergraph is a dictionary with the following keys:
        - hypergraph : a dictionary with the hyperedges as values
        - features : a dictionary with the features of the nodes as values
        - labels : a dictionary with the labels of the nodes as values
        - n : the number of nodes in the hypergraph
    """

    return {
        "hypergraph": hypergraph,
        "features": X.numpy().reshape(-1, 1),
        "labels": (
            y.numpy() if isinstance(y, torch.Tensor) else y
        ),  # this is now a graph level label
        "n": X.shape[0],  # this is the number of nodes
    }


def _process_graph(graph, dataset_type) -> tuple[dict, str]:
    """Process a single graph and return its conversion with dataset type."""
    return (_convert_to_hypergraph(graph), dataset_type)


def load_and_convert_lrgb_datasets(
    base_path: str, dataset_name: str
) -> tuple[list, list, list, list]:
    """Loads and converts LRGB datasets into the required format.

    THIS IS SPECIFICALLY FOR THE LRGB DATASETS

    Args:
        base_path:
            Path to the LRGB datasets directory
        dataset_name:
            Name of the dataset (e.g., 'peptidesstruct')

    Returns:
        List of converted Data objects
    """
    print(f"Loading {dataset_name} from {base_path}...")
    # Load train, val, and test sets
    train_data = torch.load(os.path.join(base_path, dataset_name, "train.pt"))
    val_data = torch.load(os.path.join(base_path, dataset_name, "val.pt"))
    test_data = torch.load(os.path.join(base_path, dataset_name, "test.pt"))

    print(f"Train data length: {len(train_data)}")
    print(f"Val data length: {len(val_data)}")
    print(f"Test data length: {len(test_data)}")
    # print the first few elements
    if False:
        print(f"Train data first few elements: {train_data[:5]}")
        print(f"Val data first few elements: {val_data[:5]}")
        print(f"Test data first few elements: {test_data[:5]}")
    print("*" * 100)
    print(f"the type is {type(train_data[0])}")
    print("*" * 100)
    print(train_data[0])
    print(len(train_data[0]))
    # Convert all datasets
    converted_data = []
    train_converted = []
    val_converted = []
    test_converted = []

    # Create process pool
    with mp.Pool() as pool:
        # Prepare all graph processing tasks
        tasks = []
        tasks.extend([(graph, "train") for graph in train_data])
        tasks.extend([(graph, "val") for graph in val_data])
        tasks.extend([(graph, "test") for graph in test_data])

        # Process all graphs in parallel
        results = pool.starmap(_process_graph, tasks)

        # Combine results
        for converted, dataset_type in results:
            converted_data.append(converted)
            if dataset_type == "train":
                train_converted.append(converted)
            elif dataset_type == "val":
                val_converted.append(converted)
            elif dataset_type == "test":
                test_converted.append(converted)

    return converted_data, train_converted, val_converted, test_converted


# Run
if __name__ == "__main__":
    lukas = False
    if lukas:
        data_type = "hypergraph_classification_datasets"
        # dataset_name = "reddit_hypergraphs"  # does not matter

        # Creates an instance of the encodings_saver class
        encodings_saver_instance = encodings_saver(data_type)
        # parse calls load_data
        parsed_data = encodings_saver_instance.compute_encodings()

    cluster = True
    if cluster:
        base_path = "/n/holyscratch01/mweber_lab/lrgb_datasets"
        base_path = (
            "/Users/pellegrinraphael/Desktop/Repos_GNN/Hypergraph_Encodings/data"
        )
        datasets = [
            "peptidesstruct",
        ]

        for dataset_name in datasets:
            print(f"Processing {dataset_name}...")
            (
                converted_dataset,
                train_converted,
                val_converted,
                test_converted,
            ) = load_and_convert_lrgb_datasets(base_path, dataset_name)

            # Save the converted dataset
            save_path = os.path.join("data", "graph_classification_datasets")
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving to {save_path}")

            save_file_path: str = os.path.join(
                save_path, f"{dataset_name}_hypergraphs.pickle"
            )
            with open(save_file_path, "wb") as handle:
                pickle.dump(converted_dataset, handle)
            print(f"Saved {len(converted_dataset)} graphs to {save_file_path}")

            save_file_path: str = os.path.join(
                save_path, f"{dataset_name}_hypergraphs_train.pickle"
            )
            with open(save_file_path, "wb") as handle:
                pickle.dump(train_converted, handle)
            print(f"Saved {len(train_converted)} graphs to {save_file_path}")

            save_file_path: str = os.path.join(
                save_path, f"{dataset_name}_hypergraphs_val.pickle"
            )
            with open(save_file_path, "wb") as handle:
                pickle.dump(val_converted, handle)
            print(f"Saved {len(val_converted)} graphs to {save_file_path}")

            save_file_path: str = os.path.join(
                save_path, f"{dataset_name}_hypergraphs_test.pickle"
            )
            with open(save_file_path, "wb") as handle:
                pickle.dump(test_converted, handle)
            print(f"Saved {len(test_converted)} graphs to {save_file_path}")

        # Create encoder instance and compute encodings
        encoder = encodings_saver_lrgb(dataset_name)
        results = encoder.compute_encodings(
            (converted_dataset, train_converted, val_converted, test_converted),
        )
