#!/bin/bash
#SBATCH --job-name=general_parallel       
#SBATCH --array=0-999%10           # Run 1000 jobs, max 5 concurrent
#SBATCH --time=168:00:00         
#SBATCH --mem=16GB               
#SBATCH --output=sbatch_logs/general_parallel_%A_%a.log  # %A is job ID, %a is array index
#SBATCH --partition=mweber_gpu     
#SBATCH --gpus=1                   # One GPU per task

# This is the module I use to run
# This is the most up to data bash script

# Load modules and activate environment
module load anaconda/2023.07
source activate hgencodings_gpu_weber

# Define parameters
nlayers=(2 16 32)  # Add layer counts here
add_encodings=(False True)
do_transformer=(False True)
transformer_versions=("v1" "v2")
transformer_depths=(1 2 3 10)
models=("UniGCN" "UniSAGE" "UniGCNII" "UniGAT" "UniGIN")
data_types=("cocitation" "coauthorship")
coauthorship_datasets=("cora")
cocitation_datasets=("cora" "citeseer" "pubmed")
encodings=("LDP" "Laplacian" "RW" "LCP")
rw_types=("EE" "EN" "WE")
curvature_types=("FRC")
laplacian_types=("Hodge" "Normalized")

# Create combinations array
combinations=()
for nlayer in "${nlayers[@]}"; do
    for transformer in "${do_transformer[@]}"; do
        if [ "$transformer" == "True" ]; then
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
                            else
                                datasets=("${cocitation_datasets[@]}")
                            fi
                            for dataset in "${datasets[@]}"; do
                                for add_encoding in "${add_encodings[@]}"; do
                                    if [ "$add_encoding" == "True" ]; then
                                        for encoding in "${encodings[@]}"; do
                                            if [ "$encoding" == "RW" ]; then
                                                for rw_type in "${rw_types[@]}"; do
                                                    combinations+=("$model $data_type $dataset $encoding $rw_type $transformer $transformer_version $transformer_depth $add_encoding $nlayer")
                                                done
                                            elif [ "$encoding" == "LCP" ]; then
                                                for curvature_type in "${curvature_types[@]}"; do
                                                    combinations+=("$model $data_type $dataset $encoding $curvature_type $transformer $transformer_version $transformer_depth $add_encoding $nlayer")
                                                done
                                            elif [ "$encoding" == "Laplacian" ]; then
                                                for laplacian_type in "${laplacian_types[@]}"; do
                                                    combinations+=("$model $data_type $dataset $encoding $laplacian_type $transformer $transformer_version $transformer_depth $add_encoding $nlayer")
                                                done
                                            else
                                                combinations+=("$model $data_type $dataset $encoding none $transformer $transformer_version $transformer_depth $add_encoding $nlayer")
                                            fi
                                        done
                                    else
                                        combinations+=("$model $data_type $dataset none none $transformer $transformer_version $transformer_depth $add_encoding $nlayer")
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        else
            # If transformer is disabled, use single combination without transformer parameters
            for model in "${models[@]}"; do
                for data_type in "${data_types[@]}"; do
                    if [ "$data_type" == "coauthorship" ]; then
                        datasets=("${coauthorship_datasets[@]}")
                    else
                        datasets=("${cocitation_datasets[@]}")
                    fi
                    for dataset in "${datasets[@]}"; do
                        for add_encoding in "${add_encodings[@]}"; do
                            if [ "$add_encoding" == "True" ]; then
                                for encoding in "${encodings[@]}"; do
                                    if [ "$encoding" == "RW" ]; then
                                        for rw_type in "${rw_types[@]}"; do
                                            combinations+=("$model $data_type $dataset $encoding $rw_type $transformer none none $add_encoding $nlayer")
                                        done
                                    elif [ "$encoding" == "LCP" ]; then
                                        for curvature_type in "${curvature_types[@]}"; do
                                            combinations+=("$model $data_type $dataset $encoding $curvature_type $transformer none none $add_encoding $nlayer")
                                        done
                                    elif [ "$encoding" == "Laplacian" ]; then
                                        for laplacian_type in "${laplacian_types[@]}"; do
                                            combinations+=("$model $data_type $dataset $encoding $laplacian_type $transformer none none $add_encoding $nlayer")
                                        done
                                    else
                                        combinations+=("$model $data_type $dataset $encoding none $transformer none none $add_encoding $nlayer")
                                    fi
                                done
                            else
                                combinations+=("$model $data_type $dataset none none $transformer none none $add_encoding $nlayer")
                            fi
                        done
                    done
                done
            done
        fi
    done
done

# Get the current combination based on SLURM_ARRAY_TASK_ID
combination=(${combinations[$SLURM_ARRAY_TASK_ID]})
model=${combination[0]}
data_type=${combination[1]}
dataset=${combination[2]}
encoding=${combination[3]}
encoding_type=${combination[4]}
transformer=${combination[5]}
transformer_version=${combination[6]}
transformer_depth=${combination[7]}
add_encoding=${combination[8]}
nlayer=${combination[9]}

# Create log directory
log_dir="logs_loops_general_new"
mkdir -p "$log_dir"

# Construct log filename
log_file="$log_dir/${model}_${data_type}_${dataset}"
if [ "$encoding" != "none" ]; then
    log_file="${log_file}_${encoding}"
    if [ "$encoding_type" != "none" ]; then
        log_file="${log_file}_${encoding_type}"
    fi
else
    log_file="${log_file}_no_encodings"
fi
# Only add transformer details to filename if transformer is enabled
if [ "$transformer" == "True" ]; then
    log_file="${log_file}_transformer${transformer}_${transformer_version}_depth${transformer_depth}"
fi
log_file="${log_file}_nlayer${nlayer}.log"

# Run the specific combination
echo "Running combination $SLURM_ARRAY_TASK_ID: $model $data_type $dataset $encoding $encoding_type $transformer $transformer_version $transformer_depth $add_encoding $nlayer"
python scripts/train_val.py \
    --add-self-loop \
    --model="$model" \
    --data="$data_type" \
    --dataset="$dataset" \
    --encodings="$encoding" \
    --nlayer="$nlayer" \
    $([ "$add_encoding" == "True" ] && echo "--add-encodings") \
    $([ "$add_encoding" == "False" ] && echo "--no-add-encodings") \
    $([ "$encoding" == "RW" ] && echo "--random-walk-type=$encoding_type") \
    $([ "$encoding" == "LCP" ] && echo "--curvature-type=$encoding_type") \
    $([ "$encoding" == "Laplacian" ] && echo "--laplacian-type=$encoding_type") \
    $([ "$transformer" == "True" ] && echo "--do-transformer") \
    $([ "$transformer" == "False" ] && echo "--no-transformer") \
    $([ "$transformer" == "True" ] && echo "--transformer-version=$transformer_version") \
    $([ "$transformer" == "True" ] && echo "--transformer-depth=$transformer_depth") > "$log_file" 2>&1