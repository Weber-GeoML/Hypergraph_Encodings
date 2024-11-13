#!/bin/bash
#SBATCH --job-name=ablationII       # Job name
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --time=168:00:00         # Time limit (hh:mm:ss)
#SBATCH --mem=16GB               # Memory required
#SBATCH --output=outputIIablation_%j.log  # Standard output and error log (with job ID)
#SBATCH --partition=mweber_gpu     # Specify the partition
#SBATCH --gpus=4                   # Request 1 GPU

# Load required modules (adjust based on your cluster setup)
# Load Conda (if needed)
module load anaconda/2023.07  # Example, depending on your system

# Activate the Conda environment
source activate hgencodings_gpu_weber

add_encodings=(False True)
do_transformer=(True False)
transformer_versions=("v1" "v2")  # Added transformer versions
transformer_depths=(1 2 4 8)      # Added transformer depths
models=("UniGCN" "UniSAGE" "UniGCNII")
data_types=("cocitation" "coauthorship")
coauthorship_datasets=("cora" "dblp")
cocitation_datasets=("cora" "citeseer" "pubmed")
encodings=("LDP")
rw_types=("EE" "EN" "WE")
curvature_types=("FRC")
laplacian_types=("Hodge" "Normalized")

# Create a directory to store logs
log_dir="logs_loops_general_new"
mkdir -p "$log_dir"

# Loop over all combinations
for transformer in "${do_transformer[@]}"; do
    for transformer_version in "${transformer_versions[@]}"; do
        for transformer_depth in "${transformer_depths[@]}"; do
            # Skip depth > 1 for v1
            if [ "$transformer_version" == "v1" ] && [ "$transformer_depth" -gt 1 ]; then
                continue
            fi
            for model in "${models[@]}"; do
                for data_type in "${data_types[@]}"; do
                    if [ "$data_type" == "coauthorship" ]; then
                        datasets=("${coauthorship_datasets[@]}")
                    elif [ "$data_type" == "cocitation" ]; then
                        datasets=("${cocitation_datasets[@]}")
                    fi

                    for dataset in "${datasets[@]}"; do
                        for add_encoding in "${add_encodings[@]}"; do
                            if [ "$add_encoding" == "True" ]; then
                                for encoding in "${encodings[@]}"; do
                                    if [ "$encoding" == "RW" ]; then
                                        for rw_type in "${rw_types[@]}"; do
                                            log_file="$log_dir/${model}_${data_type}_${dataset}_${encoding}_${rw_type}_transformer${transformer}_${transformer_version}_depth${transformer_depth}.log"
                                            echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding --random-walk-type=$rw_type --do-transformer=$transformer --transformer-version=$transformer_version --transformer-depth=$transformer_depth"
                                            python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding" --random-walk-type="$rw_type" --do-transformer="$transformer" --transformer-version="$transformer_version" --transformer-depth="$transformer_depth" > "$log_file" 2>&1
                                        done
                                    elif [ "$encoding" == "LCP" ]; then
                                        for curvature_type in "${curvature_types[@]}"; do
                                            log_file="$log_dir/${model}_${data_type}_${dataset}_${encoding}_${curvature_type}_transformer${transformer}_${transformer_version}_depth${transformer_depth}.log"
                                            echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding --curvature-type=$curvature_type --do-transformer=$transformer --transformer-version=$transformer_version --transformer-depth=$transformer_depth"
                                            python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding" --curvature-type="$curvature_type" --do-transformer="$transformer" --transformer-version="$transformer_version" --transformer-depth="$transformer_depth" > "$log_file" 2>&1
                                        done
                                    elif [ "$encoding" == "Laplacian" ]; then
                                        for laplacian_type in "${laplacian_types[@]}"; do
                                            log_file="$log_dir/${model}_${data_type}_${dataset}_${encoding}_${laplacian_type}_transformer${transformer}_${transformer_version}_depth${transformer_depth}.log"
                                            echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding --laplacian-type=$laplacian_type --do-transformer=$transformer --transformer-version=$transformer_version --transformer-depth=$transformer_depth"
                                            python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding" --laplacian-type="$laplacian_type" --do-transformer="$transformer" --transformer-version="$transformer_version" --transformer-depth="$transformer_depth" > "$log_file" 2>&1
                                        done
                                    else
                                        log_file="$log_dir/${model}_${data_type}_${dataset}_${encoding}_transformer${transformer}_${transformer_version}_depth${transformer_depth}.log"
                                        echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding --do-transformer=$transformer --transformer-version=$transformer_version --transformer-depth=$transformer_depth"
                                        python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding" --do-transformer="$transformer" --transformer-version="$transformer_version" --transformer-depth="$transformer_depth" > "$log_file" 2>&1
                                    fi
                                done
                            else
                                log_file="$log_dir/${model}_${data_type}_${dataset}_noencodings_transformer${transformer}_${transformer_version}_depth${transformer_depth}.log"
                                echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --add-encodings=False --do-transformer=$transformer --transformer-version=$transformer_version --transformer-depth=$transformer_depth"
                                python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --add-encodings=False --do-transformer="$transformer" --transformer-version="$transformer_version" --transformer-depth="$transformer_depth" > "$log_file" 2>&1
                            fi
                        done
                    done
                done
            done
        done
    done
done
