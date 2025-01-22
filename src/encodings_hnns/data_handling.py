import inspect
import os
import pickle
import warnings


# import hypernetx as hnx
import numpy as np

# necessary for pickle.load

warnings.simplefilter("ignore")


# Used in train_val.py
def load(args) -> tuple[dict[dict, np.matrix, np.ndarray, int], list, list]:
    """Parses the dataset

    Args:
        args:
            an object with attributes data, dataset and splits

    Returns:
        dataset:
            a dict with hypergraph, features, labels, n
        train:
            indices of train nodes
        test:
            indices of test nodes

        the len of train and test sums to the number of nodes.
    """
    print(f"The split is {args.split}")
    dataset = parser(args.data, args.dataset).parse()

    current: str = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir: str
    Dir, _ = os.path.split(current)
    Dir = os.path.dirname(os.path.dirname(Dir))
    file: str = os.path.join(
        Dir,
        "data",
        args.data,
        args.dataset,
        "splits",
        str(args.split) + ".pickle",
    )

    if not os.path.isfile(file):
        print("split + ", str(args.split), "does not exist")
    with open(file, "rb") as H:
        Splits = pickle.load(H)
        train, test = Splits["train"], Splits["test"]

    return dataset, train, test


class parser(object):
    """Parses data"""

    def __init__(self, data: str, dataset: str) -> None:
        """Initialises the data directory

        Arguments:
            data:
                coauthorship/cocitation
            dataset:
                cora/dblp/acm for coauthorship and cora/citeseer/pubmed for cocitation
        """

        current: str = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        current = os.path.dirname(os.path.dirname(current))
        # Makes the path
        if data == "coauthorship" or data == "cocitation":
            self.d: str = os.path.join(current, "data", data, dataset)
        else:
            self.d: str = os.path.join(current, "data", data)
        self.data, self.dataset = data, dataset

    def parse(self):
        """Returns a dataset specific function to parse

        Returns:
            TODO
        """

        name: str = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()

    def _load_data(self, verbose: bool = True) -> dict:
        """Loads the coauthorship hypergraph, features, and labels

        assumes the following files to be present in the dataset directory:
        hypergraph.pickle: coauthorship hypergraph
        features.pickle: bag of word features
        labels.pickle: labels of papers

        n: number of hypernodes
        returns: a dictionary with hypergraph, features, and labels
        and number of features as keys
        """

        # loads the hypergraph (and only the hypergraph, as a dict)
        with open(os.path.join(self.d, "hypergraph.pickle"), "rb") as handle:
            hypergraph: dict = pickle.load(handle)
            print("number of hyperedges is", len(hypergraph))

        # loads the features, an np.matrix
        with open(os.path.join(self.d, "features.pickle"), "rb") as handle:
            features: np.matrix = pickle.load(handle).todense()

        # loads the labels
        with open(os.path.join(self.d, "labels.pickle"), "rb") as handle:
            labels: np.array[int] = self._1hot(pickle.load(handle))

        if verbose:
            total_length: int = sum(len(value) for value in hypergraph.values())
            average_length: float = total_length / len(hypergraph)
            print(
                f"The hypergraph {self.dataset}/{self.data} has {len(hypergraph)} hyperedges where authors are hyperedges"
            )
            print(f"The average hyperedge contains {average_length} nodes")

        return {
            "hypergraph": hypergraph,
            "features": features,
            "labels": labels,
            "n": features.shape[0],  # one-hot encoded
        }

    def _1hot(self, labels: list) -> np.ndarray[int]:
        """converts each positive integer (representing a unique class) into ints one-hot form

        Arguments:
            labels:
                a list of positive integers with eah integer representing a unique label

        Returns:

        """

        classes: set = set(labels)  # the set of unique labels
        onehot: dict = {
            c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
        }
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)


# Example utilization
if __name__ == "__main__":

    print("EXAMPLE UTILIZATION")
    data_type = "coauthorship"
    dataset_name = "cora"

    # Creates an instance of the parser class
    parser_instance = parser(data_type, dataset_name)
    # parse calls load_data
    parsed_data = parser_instance.parse()
    print(parsed_data)
    # same as
    # data = parser_instance._load_data()
    # So hypergraph is a dict:
    # key: authors, values: papers participates in.
    print("hypergraph")
    # print(parsed_data["hypergraph"])
    # hgh = list(parsed_data["hypergraph"].items())
    # hnx.drawing.draw(hnx.Hypergraph(hgh))
    # plt.show()
    print("features")
    # print(parsed_data["features"])
    print("features shape")
    print(parsed_data["features"].shape)
    print("labels shape")
    print(parsed_data["labels"].shape)
    print(parsed_data["labels"][0])

    # Number of unique nodes
    all_nodes = set()
    for nodes in parsed_data["hypergraph"].values():
        all_nodes.update(nodes)

    num_nodes = len(all_nodes)
    min_node = min(all_nodes)
    max_node = max(all_nodes)

    print(f"Number of unique nodes: {num_nodes}")
    print(parsed_data["n"])
    print(min_node)
    print(max_node)
    complete_set = set(range(max_node))
    # Find the difference: elements in complete_set but not in all_nodes
    missing_nodes = complete_set - all_nodes

    print(f"Nodes in the complete set but not in all_nodes: {missing_nodes}")
    # Number of key-value pairs
    num_hyperedges = len(parsed_data["hypergraph"])
    print(f"Number of hyperedges (key-value pairs): {num_hyperedges}")

    ###### Cora
    data_type = "coauthorship"
    dataset_name = "cora"

    # Creates an instance of the parser class
    parser_instance = parser(data_type, dataset_name)
    # parse calls load_data
    parsed_data = parser_instance.parse()
    print(parsed_data["hypergraph"]["V Gupta"])
    print(parsed_data["features"][0])
    print(parsed_data["labels"][0])

    print("Alternatively calling the load function directly")

    # Create an args object with necessary attributes
    class Args:
        data = "coauthorship"
        dataset = "cora"
        split = 1

    args = Args()
    print(args)

    # Call the load function
    dataset, train, test = load(args)

    # Now you can use dataset, train, and test
    # print(train)
    # print(test)
    print("DONE")
