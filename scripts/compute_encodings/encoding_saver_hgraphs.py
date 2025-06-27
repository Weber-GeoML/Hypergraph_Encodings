"""Class for saving the encodings (not LRGB datasets)."""

import inspect
import os
import pickle
import warnings
from typing import Any, Callable, Dict, List

import torch

from base_class import EncodingsSaverBase

warnings.simplefilter("ignore")


class EncodingsSaver(EncodingsSaverBase):
    """Parses data for Protein/Enzyme/Mutag/IMDB/Collab/Reddit (not LRGB datasets)."""

    def __init__(self, data: str) -> None:
        super().__init__("hypergraph_classification_datasets")
        self.data = data

    def compute_encodings(self) -> Dict[str, Any]:
        """
        Returns a dataset-specific function to compute the encodings on the data.

        Returns:
            Dictionary of computed encodings.
        """
        name: str = "_compute_encodings"
        function: Callable[[], Any] = getattr(self, name, lambda: {})
        return function()

    def _compute_encodings(self) -> Dict[str, Any]:
        """
        Computes the encodings on the data.


        Returns:
            Dictionary of all results.
        """
        list_files: List[str] = [
            "proteins_hypergraphs",
            "enzymes_hypergraphs",
            "mutag_hypergraphs",
            "imdb_hypergraphs",
            "collab_hypergraphs",
            "reddit_hypergraphs",
        ]

        all_results: Dict[str, Any] = {}
        for lukas_file in list_files:
            results = self._process_file(lukas_file)
            all_results[lukas_file] = results

        return all_results
