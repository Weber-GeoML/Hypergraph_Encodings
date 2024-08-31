#!/bin/bash

# Define arrays for the models, datasets, and encodings
models=("UniGIN" "UniGAT" "UniGCN")
datasets=("cora" "dblp")

# Create a directory to store logs
log_dir="logs_reg"
mkdir -p "$log_dir"

# Loop over all combinations of models, datasets, and encodings
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        log_file="$log_dir/${model}_${dataset}.log"
        echo "Running: python scripts/train_val.py --model=$model --dataset=$dataset"
        python scripts/train_val.py --add-self-loop --model="$model" --dataset="$dataset" > "$log_file" 2>&1
    done
done


# do UNigin, from citrseer