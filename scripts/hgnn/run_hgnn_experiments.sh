#!/bin/bash

# Run HGNN experiments compatible with UniGNN methodology

echo "Starting HGNN UniGNN-Compatible Experiments"
echo "============================================"


# Run experiments for different datasets
python scripts/hgnn/hgnn_unignn_compatible.py \
    --data cocitation \
    --dataset cora \
    --n_runs 1 \
    --epochs 500 \
    --patience 50

echo "Experiments completed!" 