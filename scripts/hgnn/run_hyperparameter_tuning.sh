#!/bin/bash
#SBATCH --job-name=hgnn_tuning        # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --time=72:00:00              # Time limit (hh:mm:ss) - 3 days
#SBATCH --mem=32GB                   # Memory required
#SBATCH --output=logs_hgnn/tuning_%j.log  # Standard output and error log (with job ID)
#SBATCH --partition=mweber_gpu       # Specify the partition
#SBATCH --gpus=1                     # Request 1 GPU

# Load required modules (adjust based on your cluster setup)
module load anaconda/2023.07  # Example, depending on your system

# Activate the Conda environment
source activate hgencodings_gpu_weber

# Create logs directory
mkdir -p logs_hgnn

# Function to run hyperparameter tuning for a single dataset/encoding combination
run_single_tuning() {
    local data_type=$1
    local dataset_name=$2
    local encoding_type=$3
    local log_file="logs_hgnn/tuning_${data_type}_${dataset_name}_${encoding_type}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "============================================" | tee -a "$log_file"
    echo "Hyperparameter tuning: ${data_type}/${dataset_name} with ${encoding_type}" | tee -a "$log_file"
    echo "Log file: $log_file" | tee -a "$log_file"
    echo "============================================" | tee -a "$log_file"
    echo "Start time: $(date)" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    
    # Run hyperparameter tuning
    python run_hyperparameter_tuning.py \
        --data_type "$data_type" \
        --dataset_name "$dataset_name" \
        --encoding_type "$encoding_type" 2>&1 | tee -a "$log_file"
    
    echo "Completed tuning for ${data_type}/${dataset_name} with ${encoding_type}" | tee -a "$log_file"
    echo "End time: $(date)" | tee -a "$log_file"
    echo "============================================" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
}

# Main execution
echo "Starting HGNN Hyperparameter Tuning" | tee logs_hgnn/tuning_main_$(date +%Y%m%d_%H%M%S).log
echo "Total start time: $(date)" | tee logs_hgnn/tuning_main_$(date +%Y%m%d_%H%M%S).log
echo "============================================" | tee logs_hgnn/tuning_main_$(date +%Y%m%d_%H%M%S).log

# Define key encodings to tune (most important ones)
key_encodings=("none" "degree" "random_walk_EE" "curvature_ORC")

# Run hyperparameter tuning for coauthorship datasets
echo "Running hyperparameter tuning for coauthorship datasets..." | tee logs_hgnn/tuning_main_$(date +%Y%m%d_%H%M%S).log
for dataset in "cora" "dblp"; do
    for encoding in "${key_encodings[@]}"; do
        run_single_tuning "coauthorship" "$dataset" "$encoding"
    done
done

# Run hyperparameter tuning for cocitation datasets
echo "Running hyperparameter tuning for cocitation datasets..." | tee logs_hgnn/tuning_main_$(date +%Y%m%d_%H%M%S).log
for dataset in "citeseer" "cora" "pubmed"; do
    for encoding in "${key_encodings[@]}"; do
        run_single_tuning "cocitation" "$dataset" "$encoding"
    done
done

echo "All hyperparameter tuning completed!" | tee logs_hgnn/tuning_main_$(date +%Y%m%d_%H%M%S).log
echo "Total end time: $(date)" | tee logs_hgnn/tuning_main_$(date +%Y%m%d_%H%M%S).log
echo "Check the results files to find optimal hyperparameters for each dataset/encoding combination." | tee logs_hgnn/tuning_main_$(date +%Y%m%d_%H%M%S).log
