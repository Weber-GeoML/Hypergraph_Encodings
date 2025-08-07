"""Best hyperparameters for each dataset/encoding combination from tuning results."""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class BestHyperparameters:
    """Best hyperparameters for a specific dataset/encoding combination."""

    hidden_dims: int
    dropout_rate: float
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int
    val_ratio: float
    normalize_features: bool
    normalize_encodings: bool
    accuracy: float
    std: float


# Best hyperparameters from tuning results
BEST_HYPERPARAMETERS: Dict[str, BestHyperparameters] = {
    "cocitation_cora_none": BestHyperparameters(
        hidden_dims=128.0,
        dropout_rate=0.5,
        learning_rate=0.01,
        weight_decay=0.0005,
        epochs=500.0,
        patience=50.0,
        val_ratio=0.2,
        normalize_features=False,
        normalize_encodings=False,
        accuracy=60.3406,
        std=1.3142,
    ),
    "cocitation_citeseer_none": BestHyperparameters(
        hidden_dims=128.0,
        dropout_rate=0.5,
        learning_rate=0.01,
        weight_decay=0.0005,
        epochs=500.0,
        patience=50.0,
        val_ratio=0.2,
        normalize_features=False,
        normalize_encodings=False,
        accuracy=58.2953,
        std=1.4767,
    ),
    "coauthorship_cora_none": BestHyperparameters(
        hidden_dims=128.0,
        dropout_rate=0.5,
        learning_rate=0.01,
        weight_decay=0.0005,
        epochs=500.0,
        patience=50.0,
        val_ratio=0.2,
        normalize_features=False,
        normalize_encodings=False,
        accuracy=60.3406,
        std=1.3142,
    ),
}


def get_best_hyperparameters(
    data_type: str, dataset: str, encoding: str
) -> BestHyperparameters:
    """Get the best hyperparameters for a specific dataset/encoding combination."""
    key = f"{data_type}_{dataset}_{encoding}"

    if key not in BEST_HYPERPARAMETERS:
        # Return default parameters if not found
        print(f"Warning: No best hyperparameters found for {key}, using defaults")
        return BestHyperparameters(
            hidden_dims=128,
            dropout_rate=0.5,
            learning_rate=0.01,
            weight_decay=0.0005,
            epochs=500,
            patience=50,
            val_ratio=0.2,
            normalize_features=False,
            normalize_encodings=False,
            accuracy=0.0,
            std=0.0,
        )

    return BEST_HYPERPARAMETERS[key]


def update_best_hyperparameters(
    data_type: str, dataset: str, encoding: str, accuracy: float, std: float, **kwargs
) -> None:
    """Update the best hyperparameters with new tuning results."""
    key = f"{data_type}_{dataset}_{encoding}"

    # Create new best hyperparameters
    best_params = BestHyperparameters(accuracy=accuracy, std=std, **kwargs)

    BEST_HYPERPARAMETERS[key] = best_params
    print(
        f"Updated best hyperparameters for {key}: accuracy={accuracy:.4f} ± {std:.4f}"
    )
