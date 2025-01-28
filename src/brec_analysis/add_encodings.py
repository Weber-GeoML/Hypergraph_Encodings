"""Functions for adding encodings to a graph or hypergraph"""

from torch_geometric.data import Data

from encodings_hnns.encodings import HypergraphEncodings


def get_encodings(
    hg: Data,
    encoder: HypergraphEncodings,
    name_of_encoding: str,
    k_rwpe: int = 1,
    k_lape: int = 1,
) -> dict | None:
    """Helper function to get the appropriate encodings based on type.

    Args:
        hg:
            the hypergraph
        encoder:
            the encoder
        name_of_encoding:
            the name of the encoding
        k_rwpe:
            the k value for the random walk encodings
        k_lape:
            the k value for the Laplacian encodings

    Returns:
        the encodings
    """
    if name_of_encoding == "LDP":
        return encoder.add_degree_encodings(hg.copy(), verbose=False)
    elif name_of_encoding == "RWPE":
        # Add k to the name for random walks
        name_of_encoding = f"RWPE-k{k_rwpe}"
        # print(f"Adding random walk encodings with k={k_rwpe} for {name_of_encoding}")
        # print(f"features: \n {hg['features']}")
        return encoder.add_randowm_walks_encodings(
            hg.copy(),
            rw_type="WE",
            verbose=False,
            k=k_rwpe
            + 1,  # because right now our implementation if for eg k 2, do 0 and 1 hop
        )
    elif name_of_encoding == "LCP-ORC":
        return encoder.add_curvature_encodings(hg.copy(), verbose=False, type="ORC")
    elif name_of_encoding == "LCP-FRC":
        return encoder.add_curvature_encodings(hg.copy(), verbose=False, type="FRC")
    elif name_of_encoding == "LAPE-Normalized":
        return encoder.add_laplacian_encodings(
            hg.copy(), type="Normalized", verbose=False, use_same_sign=True, k=k_lape
        )
    elif name_of_encoding == "LAPE-RW":
        # Add k to the name for random walk Laplacian
        name_of_encoding = f"LAPE-RW-k{k_lape}"
        return encoder.add_laplacian_encodings(
            hg.copy(), type="RW", verbose=False, k=k_lape
        )
    elif name_of_encoding == "LAPE-Hodge":
        return encoder.add_laplacian_encodings(
            hg.copy(), type="Hodge", verbose=False, use_same_sign=True, k=k_lape
        )
    return None
