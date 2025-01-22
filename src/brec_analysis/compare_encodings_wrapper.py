"""Wrapper for comparing encodings between two (hyper)graphs"""

import os

from torch_geometric.data import Data

from brec_analysis.check_encodings_same import checks_encodings
from encodings_hnns.encodings import HypergraphEncodings


def compare_encodings(
    hg1: Data,
    hg2: Data,
    pair_idx: str | int,
    category: str,
    is_isomorphic: bool,
    level: str = "graph",
    node_mapping: dict | None = None,
) -> None:
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
    """
    encoder1 = HypergraphEncodings()
    encoder2 = HypergraphEncodings()
    assert not is_isomorphic, "All pairs in BREC are non-isomorphic"

    # Define encodings to check
    encodings_to_check = [
        ("LDP", "Local Degree Profile", True),
        ("LCP-FRC", "Local Curvature Profile - FRC", True),
        ("RWPE", "Random Walk Encodings", True),
        ("LCP-ORC", "Local Curvature Profile - ORC", False),
        ("LAPE-Normalized", "Normalized Laplacian", True),
        ("LAPE-RW", "Random Walk Laplacian", True),
        ("LAPE-Hodge", "Hodge Laplacian", True),
    ]

    output_dir = f"results/{level}_level"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")

    with open(f"{output_dir}/pair_{pair_idx}_{category.lower()}.txt", "w") as f:
        f.write(f"Analysis for pair {pair_idx} ({category}) - {level} level\n")
        f.write(f"Isomorphic: {is_isomorphic}\n\n")

        for encoding_type, description, should_be_same in encodings_to_check:
            f.write(f"\n=== {description} ===\n")
            result = checks_encodings(
                name_of_encoding=encoding_type,
                same=should_be_same,
                hg1=hg1,
                hg2=hg2,
                encoder_shrikhande=encoder1,
                encoder_rooke=encoder2,
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
            f.write(f"Result: {'Same' if result else 'Different'}\n")
