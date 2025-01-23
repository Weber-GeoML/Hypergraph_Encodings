"""Wrapper for comparing encodings between two (hyper)graphs"""

from typing import Any

from torch_geometric.data import Data

from brec_analysis.check_encodings_same import checks_encodings
from brec_analysis.encodings_to_check import ENCODINGS_TO_CHECK
from encodings_hnns.encodings import HypergraphEncodings


def compare_encodings_wrapper(
    hg1: Data,
    hg2: Data,
    pair_idx: str | int,
    category: str,
    is_isomorphic: bool,
    level: str = "graph",
    node_mapping: dict | None = None,
    encoding_type: str = "LAPE-Normalized",
) -> dict:
    """Compare encodings between two (hyper)graphs.

    Higher level function that calls the lower level functions that do the actual comparison.

    Args:
        hg1 (Data):
            The first hypergraph.
        hg2 (Data):
            The second hypergraph.
        pair_idx (str):
            The index of the pair.
        category (str):
            The category of the pair.
        is_isomorphic (bool):
            Whether the graphs are isomorphic.
        level (str):
            The level of the comparison: graph or hypergraph.
        node_mapping (dict):
            The node mapping between the two graphs.
        encoding_type (str):
            The type of encoding to use.

    Returns:
        dict:
            The results of the comparison.

    """
    encoder1 = HypergraphEncodings()
    encoder2 = HypergraphEncodings()
    assert not is_isomorphic, "All pairs in BREC are non-isomorphic"

    results: dict[str, Any] = {
        "pair_idx": pair_idx,
        "category": category,
        "level": level,
        "is_isomorphic": is_isomorphic,
        "encodings": {},
    }

    # TODO: here I want to loop through different values of k. 2, 3, 4, 20.


    encoding_result = checks_encodings(
        name_of_encoding=encoding_type,
        hg1=hg1,
        hg2=hg2,
        encoder_number_one=encoder1,
        encoder_number_two=encoder2,
        name1="Graph A",
        name2="Graph B",
        save_plots=True,
        plot_dir=f"plots/encodings/{level}/{pair_idx}",
        pair_idx=pair_idx,
        category=category,
        is_isomorphic=is_isomorphic,
        node_mapping=node_mapping,
        graph_type=level,
    )
    results["encodings"][encoding_type] = {
        "status": encoding_result["status"],
        "scaling_factor": encoding_result.get("scaling_factor"),
        "permutation": encoding_result.get("permutation"),
    }

    return results
