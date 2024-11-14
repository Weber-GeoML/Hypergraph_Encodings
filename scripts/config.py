"""File taken from UniGCN"""

import argparse


def parse():
    p = argparse.ArgumentParser(
        "UniGNN: Unified Graph and Hypergraph Message Passing Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--do-transformer",
        action="store_true",
        help="Whether to use transformer layer after convolutions",
    )
    p.add_argument(
        "--no-transformer",
        action="store_false",
        dest="do_transformer",
        help="Do not use transformer layer after convolutions",
    )
    p.add_argument(
        "--transformer-version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Version of transformer to use (v1: simple single-layer, v2: full transformer)",
    )
    p.add_argument(
        "--transformer-depth",
        type=int,
        default=10,
        help="Number of transformer layers (only used in v2)",
    )
    p.add_argument(
        "--add-encodings",
        action="store_true",
        help="whether to add encodings to the features",
    )
    p.add_argument(
        "--no-add-encodings",
        action="store_false",
        dest="add_encodings",
        help="do not add encodings to the features",
    )
    p.add_argument(
        "--add-encodings-hg-classification",
        action="store_true",
        help="whether to add encodings to the features for hypergraph classification",
    )
    p.add_argument(
        "--no-add-encodings-hg-classification",
        action="store_false",
        dest="add_encodings_hg_classification",
        help="do not add encodings to the features for hypergraph classification",
    )
    p.add_argument(
        "--dataset-hypergraph-classification",
        type=str,
        default="imdb",
        choices=["imdb", "collab", "reddit", "proteins", "mutag", "enzymes"],
        help="Dataset to use for training",
    )
    p.add_argument(
        "--encodings",
        type=str,
        default="LCP",
        help="what encodings to add. RW, LDP, LCP, Laplacian",
    )
    p.add_argument(
        "--random-walk-type",
        type=str,
        default="WE",
        help="what random walk to use - WE, EN, EE",
    )
    p.add_argument(
        "--curvature-type",
        type=str,
        default="ORC",
        help="what curvature to use. ORC or FRC",
    )
    p.add_argument(
        "--normalize-features",
        action="store_true",
        help="whether to normalize features",
    )
    p.add_argument(
        "--no-normalize-features",
        action="store_false",
        dest="normalize_features",
        help="do not normalize features",
    )
    p.add_argument(
        "--normalize-encodings",
        action="store_true",
        help="whether to normalize encodings",
    )
    p.add_argument(
        "--no-normalize-encodings",
        action="store_false",
        dest="normalize_encodings",
        help="do not normalize encodings",
    )
    p.add_argument(
        "--laplacian-type",
        type=str,
        default="Hodge",
        help="what Laplacian to use (Hodge or Normalized)",
    )
    p.add_argument(
        "--k-rw",
        type=int,
        default=20,
        help="number of hops for random walks",
    )
    p.add_argument(
        "--data",
        type=str,
        default="cocitation",
        help="data name (coauthorship/cocitation)",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer/pubmed for cocitation)",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="UniGCN",
        help="UniGNN Model(UniGCN, UniGAT, UniGIN, UniSAGE...)",
    )
    p.add_argument(
        "--first-aggregate",
        type=str,
        default="mean",
        help="aggregation for hyperedge h_e: max, sum, mean",
    )
    p.add_argument(
        "--second-aggregate",
        type=str,
        default="sum",
        help="aggregation for node x_i: max, sum, mean",
    )
    p.add_argument(
        "--add-self-loop", action="store_true", help="add-self-loop to hypergraph"
    )
    p.add_argument(
        "--use-norm", action="store_true", help="use norm in the final layer"
    )
    p.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="activation layer between UniConvs",
    )
    p.add_argument("--nlayer", type=int, default=2, help="number of hidden layers")
    p.add_argument(
        "--nhid",
        type=int,
        default=8,
        help="number of hidden features, note that actually it's #nhid x #nhead",
    )
    p.add_argument("--nhead", type=int, default=8, help="number of conv heads")
    p.add_argument(
        "--dropout",
        type=float,
        default=0.6,
        help="dropout probability after UniConv layer",
    )
    p.add_argument(
        "--input-drop",
        type=float,
        default=0.6,
        help="dropout probability for input layer",
    )
    p.add_argument(
        "--attn-drop",
        type=float,
        default=0.6,
        help="dropout probability for attentions in UniGATConv",
    )
    p.add_argument("--lr", type=float, default=0.01, help="learning rate")
    p.add_argument("--wd", type=float, default=5e-4, help="weight decay")
    p.add_argument("--epochs", type=int, default=200, help="number of epochs to train")
    p.add_argument(
        "--n-runs", type=int, default=10, help="number of runs for repeated experiments"
    )
    p.add_argument("--gpu", type=int, default=0, help="gpu id to use")
    p.add_argument("--seed", type=int, default=1, help="seed for randomness")
    p.add_argument(
        "--patience", type=int, default=200, help="early stop after specific epochs"
    )
    p.add_argument(
        "--nostdout", action="store_true", help="do not output logging to terminal"
    )
    p.add_argument(
        "--split", type=int, default=1, help="choose which train/test split to use"
    )
    p.add_argument("--out-dir", type=str, default="runs/test", help="output dir")
    return p.parse_args()
