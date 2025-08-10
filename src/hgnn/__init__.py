"""HGNN architecture and configuration modules."""

from .hgnn_architecture import HGNN
from .hgnn_config import (
    CITATION_DATASETS,
    COAUTHORSHIP_DATASETS,
    DATASET_CONFIGS,
    ENCODING_TYPES,
    HGNNConfig,
    get_dataset_specific_configs,
    get_default_config,
    get_fast_config,
    get_hyperparameter_tuning_configs,
)

__all__ = [
    "HGNNConfig",
    "get_default_config",
    "get_fast_config",
    "get_hyperparameter_tuning_configs",
    "get_dataset_specific_configs",
    "ENCODING_TYPES",
    "DATASET_CONFIGS",
    "COAUTHORSHIP_DATASETS",
    "CITATION_DATASETS",
    "HGNN",
]
