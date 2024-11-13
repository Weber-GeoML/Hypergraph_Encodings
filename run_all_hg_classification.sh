#!/bin/bash

#SBATCH --job-name=hg_class    # Job name
#SBATCH --output=batch_%j.out  # Standard output and error log
#SBATCH --error=batch_%j.err   # Error log
#SBATCH --time=24:00:00        # Time limit hrs:min:sec
#SBATCH --nodes=1              # Number of nodes
#SBATCH --ntasks=1             # Number of tasks
#SBATCH --cpus-per-task=4      # Number of CPU cores
#SBATCH --mem=16GB             # Memory limit
#SBATCH --gres=gpu:1           # Request 1 GPU

# Load required modules (adjust based on your cluster setup)
# Load Conda (if needed)
module load anaconda/2023.07  # Example, depending on your system

# Activate the Conda environment
source activate hgencodings_gpu_weber

# Define parameters
do_transformer=(True False)
transformer_versions=("v1" "v2")  # Added transformer versions
transformer_depths=(1 2 4 8)      # Added transformer depths
add_encodings_hg=(True False)
models=("UniGCN" "UniSAGE" "UniGCNII")
datasets=("imdb" "proteins" "mutag" "collab" "reddit" "enzymes")  # all hg classification datasets
encodings=("LDP" "LCP" "Laplacian" "RW")
rw_types=("EE" "EN" "WE")
curvature_types=("FRC" "ORC")
laplacian_types=("Hodge" "Normalized")
nlayers=(2 3 4)

# Create a directory to store logs
log_dir="logs_hg_classification"
mkdir -p "$log_dir"

# Loop over all combinations
for transformer in "${do_transformer[@]}"; do
    for transformer_version in "${transformer_versions[@]}"; do
        for transformer_depth in "${transformer_depths[@]}"; do
            # Skip depth > 1 for v1
            if [ "$transformer_version" == "v1" ] && [ "$transformer_depth" -gt 1 ]; then
                continue
            fi
            for nlayer in "${nlayers[@]}"; do
                for model in "${models[@]}"; do
                    for dataset in "${datasets[@]}"; do
                        for add_encoding in "${add_encodings_hg[@]}"; do
                            if [ "$add_encoding" == "True" ]; then
                                for encoding in "${encodings[@]}"; do
                                    if [ "$encoding" == "RW" ]; then
                                        for rw_type in "${rw_types[@]}"; do
                                            log_file="$log_dir/${model}_${dataset}_${encoding}_${rw_type}_transformer${transformer}_${transformer_version}_depth${transformer_depth}_layer${nlayer}.log"
                                            echo "Running with: model=$model, dataset=$dataset, encoding=$encoding, rw_type=$rw_type, transformer=$transformer, version=$transformer_version, depth=$transformer_depth, nlayer=$nlayer"
                                            python scripts/run_hg_classification.py \
                                                --model="$model" \
                                                --dataset-hypergraph-classification="$dataset" \
                                                --encodings="$encoding" \
                                                --random-walk-type="$rw_type" \
                                                --do-transformer="$transformer" \
                                                --transformer-version="$transformer_version" \
                                                --transformer-depth="$transformer_depth" \
                                                --add-encodings-hg-classification="$add_encoding" \
                                                --nlayer="$nlayer" \
                                                --epochs=1000 > "$log_file" 2>&1
                                        done
                                    elif [ "$encoding" == "LCP" ]; then
                                        for curvature_type in "${curvature_types[@]}"; do
                                            log_file="$log_dir/${model}_${dataset}_${encoding}_${curvature_type}_transformer${transformer}_${transformer_version}_depth${transformer_depth}_layer${nlayer}.log"
                                            echo "Running with: model=$model, dataset=$dataset, encoding=$encoding, curvature=$curvature_type, transformer=$transformer, version=$transformer_version, depth=$transformer_depth, nlayer=$nlayer"
                                            python scripts/run_hg_classification.py \
                                                --model="$model" \
                                                --dataset-hypergraph-classification="$dataset" \
                                                --encodings="$encoding" \
                                                --curvature-type="$curvature_type" \
                                                --do-transformer="$transformer" \
                                                --transformer-version="$transformer_version" \
                                                --transformer-depth="$transformer_depth" \
                                                --add-encodings-hg-classification="$add_encoding" \
                                                --nlayer="$nlayer" \
                                                --epochs=1000 > "$log_file" 2>&1
                                        done
                                    elif [ "$encoding" == "Laplacian" ]; then
                                        for laplacian_type in "${laplacian_types[@]}"; do
                                            log_file="$log_dir/${model}_${dataset}_${encoding}_${laplacian_type}_transformer${transformer}_${transformer_version}_depth${transformer_depth}_layer${nlayer}.log"
                                            echo "Running with: model=$model, dataset=$dataset, encoding=$encoding, laplacian=$laplacian_type, transformer=$transformer, version=$transformer_version, depth=$transformer_depth, nlayer=$nlayer"
                                            python scripts/run_hg_classification.py \
                                                --model="$model" \
                                                --dataset-hypergraph-classification="$dataset" \
                                                --encodings="$encoding" \
                                                --laplacian-type="$laplacian_type" \
                                                --do-transformer="$transformer" \
                                                --transformer-version="$transformer_version" \
                                                --transformer-depth="$transformer_depth" \
                                                --add-encodings-hg-classification="$add_encoding" \
                                                --nlayer="$nlayer" \
                                                --epochs=1000 > "$log_file" 2>&1
                                        done
                                    else
                                        log_file="$log_dir/${model}_${dataset}_${encoding}_transformer${transformer}_${transformer_version}_depth${transformer_depth}_layer${nlayer}.log"
                                        echo "Running with: model=$model, dataset=$dataset, encoding=$encoding, transformer=$transformer, version=$transformer_version, depth=$transformer_depth, nlayer=$nlayer"
                                        python scripts/run_hg_classification.py \
                                            --model="$model" \
                                            --dataset-hypergraph-classification="$dataset" \
                                            --encodings="$encoding" \
                                            --do-transformer="$transformer" \
                                            --transformer-version="$transformer_version" \
                                            --transformer-depth="$transformer_depth" \
                                            --add-encodings-hg-classification="$add_encoding" \
                                            --nlayer="$nlayer" \
                                            --epochs=1000 > "$log_file" 2>&1
                                    fi
                                done
                            else
                                log_file="$log_dir/${model}_${dataset}_noencodings_transformer${transformer}_${transformer_version}_depth${transformer_depth}_layer${nlayer}.log"
                                echo "Running with: model=$model, dataset=$dataset, no encodings, transformer=$transformer, version=$transformer_version, depth=$transformer_depth, nlayer=$nlayer"
                                python scripts/run_hg_classification.py \
                                    --model="$model" \
                                    --dataset-hypergraph-classification="$dataset" \
                                    --add-encodings-hg-classification="$add_encoding" \
                                    --do-transformer="$transformer" \
                                    --transformer-version="$transformer_version" \
                                    --transformer-depth="$transformer_depth" \
                                    --nlayer="$nlayer" \
                                    --epochs=1000 > "$log_file" 2>&1
                            fi
                        done
                    done
                done
            done
        done
    done
done