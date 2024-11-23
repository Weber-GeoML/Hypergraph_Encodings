import warnings
from typing import Callable

from encodings_hnns.save_lukas_encodings_base_class import EncodingsSaverBase

warnings.simplefilter("ignore")


# TODO TODO TODO
# TODO: Also need to modify this file to account for the changes I made,
# THEN need something in scripts, then a bash script.


class encodings_saver(EncodingsSaverBase):
    """Parses data"""

    def __init__(self, data: str) -> None:
        super().__init__("hypergraph_classification_datasets")
        self.data = data

    def compute_encodings(
        self,
    ) -> dict[str, tuple[list, list, list, list, list, list, list, list]]:
        """Returns a dataset specific function to compute the
        encodings on the data added by Lukas

        Returns:
            a function to compute the encodings
        """

        name: str = "_compute_encodings"
        function: Callable = getattr(self, name, lambda: {})
        return function()

    def _compute_encodings(self) -> dict:
        """Computes the encodings on the data"""

        list_files: list[str] = [
            "proteins_hypergraphs",
            "enzymes_hypergraphs",
            "mutag_hypergraphs",
            "imdb_hypergraphs",
            "collab_hypergraphs",
            "reddit_hypergraphs",
        ]

        all_results: dict[
            str, tuple[list, list, list, list, list, list, list, list]
        ] = {}

        for lukas_file in list_files:
            results = self._process_file(lukas_file)
            all_results[lukas_file] = results

        return all_results
