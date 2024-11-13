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
module purge
module load cuda/11.7
module load python/3.11

# Define parameters
do_transformer=(True False)
add_encodings_hg=(True False)
models=("UniGCN" "UniSAGE" "UniGCNII")
datasets=("imdb" "proteins" "mutag" "collab" "reddit" "enzymes")  # all hg classification datasets
encodings=("LDP" "LCP" "Laplacian" "RW")
rw_types=("EE" "EN" "WE")
curvature_types=("FRC" "ORC")
laplacian_types=("Hodge" "Normalized")
nlayers=(2 3 4)  # different number of layers to try

# Create a directory to store logs
log_dir="logs_hg_classification"
mkdir -p "$log_dir"

# Loop over all combinations
for transformer in "${do_transformer[@]}"; do
    for nlayer in "${nlayers[@]}"; do
        for model in "${models[@]}"; do
            for dataset in "${datasets[@]}"; do
                for add_encoding in "${add_encodings_hg[@]}"; do
                    if [ "$add_encoding" == "True" ]; then
                        for encoding in "${encodings[@]}"; do
                            if [ "$encoding" == "RW" ]; then
                                for rw_type in "${rw_types[@]}"; do
                                    log_file="$log_dir/${model}_${dataset}_${encoding}_${rw_type}_transformer${transformer}_layer${nlayer}.log"
                                    echo "Running with: model=$model, dataset=$dataset, encoding=$encoding, rw_type=$rw_type, transformer=$transformer, nlayer=$nlayer"
                                    python scripts/run_hg_classification.py \
                                        --model="$model" \
                                        --dataset-hypergraph-classification="$dataset" \
                                        --encodings="$encoding" \
                                        --random-walk-type="$rw_type" \
                                        --do-transformer="$transformer" \
                                        --add-encodings-hg-classification="$add_encoding" \
                                        --nlayer="$nlayer" \
                                        --epochs=1000 > "$log_file" 2>&1
                                done
                            elif [ "$encoding" == "LCP" ]; then
                                for curvature_type in "${curvature_types[@]}"; do
                                    log_file="$log_dir/${model}_${dataset}_${encoding}_${curvature_type}_transformer${transformer}_layer${nlayer}.log"
                                    echo "Running with: model=$model, dataset=$dataset, encoding=$encoding, curvature=$curvature_type, transformer=$transformer, nlayer=$nlayer"
                                    python scripts/run_hg_classification.py \
                                        --model="$model" \
                                        --dataset-hypergraph-classification="$dataset" \
                                        --encodings="$encoding" \
                                        --curvature-type="$curvature_type" \
                                        --do-transformer="$transformer" \
                                        --add-encodings-hg-classification="$add_encoding" \
                                        --nlayer="$nlayer" \
                                        --epochs=1000 > "$log_file" 2>&1
                                done
                            elif [ "$encoding" == "Laplacian" ]; then
                                for laplacian_type in "${laplacian_types[@]}"; do
                                    log_file="$log_dir/${model}_${dataset}_${encoding}_${laplacian_type}_transformer${transformer}_layer${nlayer}.log"
                                    echo "Running with: model=$model, dataset=$dataset, encoding=$encoding, laplacian=$laplacian_type, transformer=$transformer, nlayer=$nlayer"
                                    python scripts/run_hg_classification.py \
                                        --model="$model" \
                                        --dataset-hypergraph-classification="$dataset" \
                                        --encodings="$encoding" \
                                        --laplacian-type="$laplacian_type" \
                                        --do-transformer="$transformer" \
                                        --add-encodings-hg-classification="$add_encoding" \
                                        --nlayer="$nlayer" \
                                        --epochs=1000 > "$log_file" 2>&1
                                done
                            else
                                log_file="$log_dir/${model}_${dataset}_${encoding}_transformer${transformer}_layer${nlayer}.log"
                                echo "Running with: model=$model, dataset=$dataset, encoding=$encoding, transformer=$transformer, nlayer=$nlayer"
                                python scripts/run_hg_classification.py \
                                    --model="$model" \
                                    --dataset-hypergraph-classification="$dataset" \
                                    --encodings="$encoding" \
                                    --do-transformer="$transformer" \
                                    --add-encodings-hg-classification="$add_encoding" \
                                    --nlayer="$nlayer" \
                                    --epochs=1000 > "$log_file" 2>&1
                            fi
                        done
                    else
                        log_file="$log_dir/${model}_${dataset}_noencodings_transformer${transformer}_layer${nlayer}.log"
                        echo "Running with: model=$model, dataset=$dataset, no encodings, transformer=$transformer, nlayer=$nlayer"
                        python scripts/run_hg_classification.py \
                            --model="$model" \
                            --dataset-hypergraph-classification="$dataset" \
                            --add-encodings-hg-classification="$add_encoding" \
                            --do-transformer="$transformer" \
                            --nlayer="$nlayer" \
                            --epochs=1000 > "$log_file" 2>&1
                    fi
                done
            done
        done
    done
done