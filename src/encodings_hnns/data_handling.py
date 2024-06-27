import inspect
import os
import pickle
import warnings

import matplotlib.pyplot as plt

# import hypernetx as hnx
import numpy as np

# necessary for pickle.load
import scipy.sparse as sp

warnings.simplefilter("ignore")


# Not used yet?
def load(args) -> tuple:
    """
    parses the dataset

    Argument:
        args:
            an object with attributes data, dataset and splits
    """
    print(f"The split is {args.split}")
    dataset = parser(args.data, args.dataset).parse()

    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir, _ = os.path.split(current)
    Dir = os.path.dirname(os.path.dirname(Dir))
    file = os.path.join(
        Dir, "data", args.data, args.dataset, "splits", str(args.split) + ".pickle"
    )

    if not os.path.isfile(file):
        print("split + ", str(args.split), "does not exist")
    with open(file, "rb") as H:
        Splits = pickle.load(H)
        train, test = Splits["train"], Splits["test"]

    return dataset, train, test


class parser(object):
    """an object for parsing data"""

    def __init__(self, data: str, dataset: str) -> None:
        """initialises the data directory

        arguments:
            data:
                coauthorship/cocitation
            dataset:
                cora/dblp/acm for coauthorship and cora/citeseer/pubmed for cocitation
        """

        current = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        current = os.path.dirname(os.path.dirname(current))
        # Makes the path
        self.d: str = os.path.join(current, "data", data, dataset)
        self.data, self.dataset = data, dataset

    def parse(self):
        """
        returns a dataset specific function to parse
        """

        name: str = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()

    def _load_data(self, verbose: bool = True) -> dict:
        """loads the coauthorship hypergraph, features, and labels of cora

        assumes the following files to be present in the dataset directory:
        hypergraph.pickle: coauthorship hypergraph
        features.pickle: bag of word features
        labels.pickle: labels of papers

        n: number of hypernodes
        returns: a dictionary with hypergraph, features, and labels
        and number of features as keys
        """

        with open(os.path.join(self.d, "hypergraph.pickle"), "rb") as handle:
            hypergraph = pickle.load(handle)
            print("number of hyperedges is", len(hypergraph))

        with open(os.path.join(self.d, "features.pickle"), "rb") as handle:
            features = pickle.load(handle).todense()

        with open(os.path.join(self.d, "labels.pickle"), "rb") as handle:
            labels = self._1hot(pickle.load(handle))

        if verbose:
            total_length = sum(len(value) for value in hypergraph.values())
            average_length = total_length / len(hypergraph)
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

    def _1hot(self, labels: list) -> np.ndarray:
        """converts each positive integer (representing a unique class) into ints one-hot form

        Arguments:
            labels:
                a list of positive integers with eah integer representing a unique label

        Returns:

        """

        classes = set(labels)
        onehot: dict = {
            c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
        }
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)


# Example utilization
if __name__ == "__main__":
    data_type = "cocitation"
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
    # print(parsed_data["hypergraph"])
    # hgh = list(parsed_data["hypergraph"].items())
    # hnx.drawing.draw(hnx.Hypergraph(hgh))
    # plt.show()
    print(parsed_data["features"][0])
    print(parsed_data["labels"][0])

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
