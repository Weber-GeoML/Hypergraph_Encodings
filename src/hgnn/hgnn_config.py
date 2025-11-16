# -*- coding: utf-8 -*-
"""Configuration file for HGNN hyperparameters and experiment settings"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class HGNNConfig:
    """Configuration class for HGNN experiments."""

    # Model hyperparameters
    hidden_dims: int = 16
    dropout_rate: float = 0.5

    # Training hyperparameters
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 500
    patience: int = 50
    val_ratio: float = 0.2

    # Experiment settings
    n_runs: int = 80  # 8 seeds × 10 runs
    n_seeds: int = 8  # Seeds 2-9
    runs_per_seed: int = 10

    # Data preprocessing
    normalize_features: bool = False
    normalize_encodings: bool = False

    # Device settings
    gpu_id: int = 0

    # Logging
    verbose: bool = True
    save_results: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "patience": self.patience,
            "val_ratio": self.val_ratio,
            "n_runs": self.n_runs,
            "n_seeds": self.n_seeds,
            "runs_per_seed": self.runs_per_seed,
            "normalize_features": self.normalize_features,
            "normalize_encodings": self.normalize_encodings,
            "gpu_id": self.gpu_id,
            "verbose": self.verbose,
            "save_results": self.save_results,
        }


# Predefined configurations for different experiment types
def get_default_config() -> HGNNConfig:
    """Get default configuration."""
    return HGNNConfig()


def get_fast_config() -> HGNNConfig:
    """Get configuration for quick testing."""
    config = HGNNConfig()
    config.epochs = 100
    config.patience = 20
    config.n_runs = 10  # Just 1 seed × 10 runs for quick testing
    config.n_seeds = 1
    return config


def get_hyperparameter_tuning_configs() -> List[HGNNConfig]:
    """Get configurations for hyperparameter tuning."""
    configs = []

    # Learning rate tuning
    for lr in [0.001, 0.005, 0.01, 0.05, 0.1]:
        config = HGNNConfig()
        config.learning_rate = lr
        config.n_runs = 20  # Fewer runs for tuning
        config.n_seeds = 2
        configs.append(config)

    # Hidden dimensions tuning
    for hidden_dim in [8, 16, 32, 64, 128]:
        config = HGNNConfig()
        config.hidden_dims = hidden_dim
        config.n_runs = 20
        config.n_seeds = 2
        configs.append(config)

    # Dropout rate tuning
    for dropout in [0.1, 0.3, 0.5, 0.7]:
        config = HGNNConfig()
        config.dropout_rate = dropout
        config.n_runs = 20
        config.n_seeds = 2
        configs.append(config)

    # Weight decay tuning
    for wd in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        config = HGNNConfig()
        config.weight_decay = wd
        config.n_runs = 20
        config.n_seeds = 2
        configs.append(config)

    return configs


def get_dataset_specific_configs() -> Dict[str, HGNNConfig]:
    """Get configurations optimized for specific datasets."""
    configs = {}

    # Cora-specific config
    cora_config = HGNNConfig()
    cora_config.hidden_dims = 32
    cora_config.learning_rate = 0.01
    cora_config.weight_decay = 5e-4
    configs["cora"] = cora_config

    # Citeseer-specific config
    citeseer_config = HGNNConfig()
    citeseer_config.hidden_dims = 64
    citeseer_config.learning_rate = 0.005
    citeseer_config.weight_decay = 1e-4
    configs["citeseer"] = citeseer_config

    # PubMed-specific config
    pubmed_config = HGNNConfig()
    pubmed_config.hidden_dims = 128
    pubmed_config.learning_rate = 0.01
    pubmed_config.weight_decay = 5e-4
    configs["pubmed"] = pubmed_config

    # DBLP-specific config
    dblp_config = HGNNConfig()
    dblp_config.hidden_dims = 64
    dblp_config.learning_rate = 0.01
    dblp_config.weight_decay = 1e-4
    configs["dblp"] = dblp_config

    return configs


# Available encoding types
ENCODING_TYPES = [
    "none",
    "degree",
    "random_walk_EE",
    "random_walk_EN",
    "random_walk_WE",
    "laplacian_Hodge",
    "laplacian_Normalized",
    "curvature_ORC",
    "curvature_FRC",
]

# Available datasets
COAUTHORSHIP_DATASETS = ["cora", "dblp"]
CITATION_DATASETS = ["citeseer", "cora", "pubmed"]

# Dataset mapping
DATASET_CONFIGS = {
    "coauthorship": COAUTHORSHIP_DATASETS,
    "cocitation": CITATION_DATASETS,
}
