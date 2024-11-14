#!/bin/bash
#SBATCH --job-name=hg_classification       
#SBATCH --array=0-999%5           # Run 1000 jobs, max 5 concurrent
#SBATCH --time=168:00:00         
#SBATCH --mem=16GB               
#SBATCH --output=hg_classification_%A_%a.log  # %A is job ID, %a is array index
#SBATCH --partition=mweber_gpu     
#SBATCH --gpus=1                   # One GPU per task

# Load modules and activate environment
module load anaconda/2023.07
source activate hgencodings_gpu_weber

# Define parameters (same as before)
do_transformer=(True False)
transformer_versions=("v1" "v2")
transformer_depths=(1 2 4 8)
add_encodings_hg=(True False)
models=("UniGCN" "UniSAGE" "UniGCNII")
datasets=("imdb" "proteins" "mutag" "collab" "reddit" "enzymes")
encodings=("LDP" "LCP" "Laplacian" "RW")
rw_types=("EE" "EN" "WE")
curvature_types=("FRC" "ORC")
laplacian_types=("Hodge" "Normalized")
nlayers=(2 3 4)

# Create combinations array
combinations=()
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
                                            combinations+=("$model $dataset $encoding $rw_type $transformer $transformer_version $transformer_depth $nlayer $add_encoding")
                                        done
                                    elif [ "$encoding" == "LCP" ]; then
                                        for curvature_type in "${curvature_types[@]}"; do
                                            combinations+=("$model $dataset $encoding $curvature_type $transformer $transformer_version $transformer_depth $nlayer $add_encoding")
                                        done
                                    elif [ "$encoding" == "Laplacian" ]; then
                                        for laplacian_type in "${laplacian_types[@]}"; do
                                            combinations+=("$model $dataset $encoding $laplacian_type $transformer $transformer_version $transformer_depth $nlayer $add_encoding")
                                        done
                                    else
                                        combinations+=("$model $dataset $encoding none $transformer $transformer_version $transformer_depth $nlayer $add_encoding")
                                    fi
                                done
                            else
                                combinations+=("$model $dataset none none $transformer $transformer_version $transformer_depth $nlayer $add_encoding")
                            fi
                        done
                    done
                done
            done
        done
    done
done

# Get the current combination based on SLURM_ARRAY_TASK_ID
combination=(${combinations[$SLURM_ARRAY_TASK_ID]})
model=${combination[0]}
dataset=${combination[1]}
encoding=${combination[2]}
encoding_type=${combination[3]}
transformer=${combination[4]}
transformer_version=${combination[5]}
transformer_depth=${combination[6]}
nlayer=${combination[7]}
add_encoding=${combination[8]}

# Create log directory
log_dir="logs_hg_classification"
mkdir -p "$log_dir"

# Construct log filename
log_file="$log_dir/${model}_${dataset}"
if [ "$encoding" != "none" ]; then
    log_file="${log_file}_${encoding}"
    if [ "$encoding_type" != "none" ]; then
        log_file="${log_file}_${encoding_type}"
    fi
fi
log_file="${log_file}_transformer${transformer}_${transformer_version}_depth${transformer_depth}_layer${nlayer}.log"

# Run the specific combination
echo "Running combination $SLURM_ARRAY_TASK_ID: $model $dataset $encoding $encoding_type $transformer $transformer_version $transformer_depth $nlayer $add_encoding"
python scripts/run_hg_classification.py \
    --model="$model" \
    --dataset-hypergraph-classification="$dataset" \
    --encodings="$encoding" \
    $([ "$encoding" == "RW" ] && echo "--random-walk-type=$encoding_type") \
    $([ "$encoding" == "LCP" ] && echo "--curvature-type=$encoding_type") \
    $([ "$encoding" == "Laplacian" ] && echo "--laplacian-type=$encoding_type") \
    --do-transformer="$transformer" \
    --transformer-version="$transformer_version" \
    --transformer-depth="$transformer_depth" \
    --add-encodings-hg-classification="$add_encoding" \
    --nlayer="$nlayer" \
    --epochs=1000 > "$log_file" 2>&1