#!/bin/bash

# Define arrays for the models, datasets, and encodings
models=("UniGCN" "UniGAT" "UniGIN")
datasets=("cora" "citeseer" "pubmed")
encodings=("LCP" "LDP" "Laplacian" "RW")

# Create a directory to store logs
log_dir="logs_loops_norm"
mkdir -p "$log_dir"

# Loop over all combinations of models, datasets, and encodings
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for encoding in "${encodings[@]}"; do
            log_file="$log_dir/${model}_${dataset}_${encoding}.log"
            echo "Running: python scripts/train_val.py --model=$model --dataset=$dataset --encodings=$encoding"
            python scripts/train_val.py --use-norm --add-self-loop --model="$model" --dataset="$dataset" --encodings="$encoding" > "$log_file" 2>&1
        done
    done
done


# do UNigin, from citrseer