# -*- coding: utf-8 -*-
"""Configuration for HGNN experiments"""

import argparse


def parse():
    """Parse command line arguments for HGNN experiments."""
    parser = argparse.ArgumentParser(description="HGNN UniGNN-Compatible Experiments")

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default="cocitation",
        choices=["cocitation", "coauthorship"],
        help="Data type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "citeseer", "pubmed", "dblp"],
        help="Dataset name",
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument(
        "--patience", type=int, default=100, help="Patience for early stopping"
    )
    parser.add_argument("--n_runs", type=int, default=1, help="Number of runs per seed")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")

    # Encoding arguments (for compatibility)
    parser.add_argument("--add_encodings", action="store_true", help="Add encodings")
    parser.add_argument("--encodings", type=str, default=None, help="Encoding type")
    parser.add_argument(
        "--normalize_features",
        action="store_true",
        default=True,
        help="Normalize features",
    )
    parser.add_argument(
        "--normalize_encodings",
        action="store_true",
        default=True,
        help="Normalize encodings",
    )

    # System arguments
    parser.add_argument("--gpu", type=int, default=0, help="GPU device")
    parser.add_argument("--split", type=int, default=1, help="Data split")

    return parser.parse_args()
