#!/bin/bash
#SBATCH --job-name=hgnn_experiments     # Job name
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --time=168:00:00                # Time limit (hh:mm:ss) - 7 days
#SBATCH --mem=32GB                      # Memory required
#SBATCH --output=logs_hgnn/hgnn_%j.log  # Standard output and error log (with job ID)
#SBATCH --partition=mweber_gpu          # Specify the partition
#SBATCH --gpus=1                        # Request 1 GPU

# Load required modules (adjust based on your cluster setup)
module load anaconda/2023.07  # Example, depending on your system

# Activate the Conda environment
source activate hgencodings_gpu_weber

# Create logs directory
mkdir -p logs_hgnn

# Define datasets for each type
coauthorship_datasets=("cora" "dblp")
cocitation_datasets=("citeseer" "cora" "pubmed")

# Define encoding types to test
encoding_types=(
    "none"
    "degree"
    "random_walk_EE"
    "random_walk_EN"
    "random_walk_WE"
    "laplacian_Hodge"
    "laplacian_Normalized"
    "curvature_ORC"
    "curvature_FRC"
)

# Function to run experiments for a single dataset
run_dataset_experiments() {
    local data_type=$1
    local dataset_name=$2
    local log_file="logs_hgnn/${data_type}_${dataset_name}_hgnn_$(date +%Y%m%d_%H%M%S).log"
    
    echo "============================================" | tee -a "$log_file"
    echo "Starting HGNN experiments for ${data_type}/${dataset_name}" | tee -a "$log_file"
    echo "Log file: $log_file" | tee -a "$log_file"
    echo "============================================" | tee -a "$log_file"
    echo "Start time: $(date)" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    
    # Run experiments for each encoding type
    for encoding_type in "${encoding_types[@]}"; do
        echo "Running experiment: ${data_type}/${dataset_name} with encoding: ${encoding_type}" | tee -a "$log_file"
        echo "Experiment start time: $(date)" | tee -a "$log_file"
        
        # Run the HGNN script with proper logging
        python scripts/hgnn/hgnn_m3.py \
            --data "$data_type" \
            --dataset "$dataset_name" \
            --encoding "$encoding_type" \
            --n_runs 80 \
            --epochs 500 \
            --patience 50 \
            --gpu 0 \
            --normalize_features \
            --normalize_encodings 2>&1 | tee -a "$log_file"
        
        echo "Experiment end time: $(date)" | tee -a "$log_file"
        echo "--------------------------------------------" | tee -a "$log_file"
        echo "" | tee -a "$log_file"
    done
    
    echo "Completed all experiments for ${data_type}/${dataset_name}" | tee -a "$log_file"
    echo "End time: $(date)" | tee -a "$log_file"
    echo "============================================" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
}

# Main execution
echo "Starting HGNN UniGNN-Compatible Experiments" | tee logs_hgnn/main_$(date +%Y%m%d_%H%M%S).log
echo "Total start time: $(date)" | tee logs_hgnn/main_$(date +%Y%m%d_%H%M%S).log
echo "============================================" | tee logs_hgnn/main_$(date +%Y%m%d_%H%M%S).log

# Run experiments for coauthorship datasets
echo "Running coauthorship datasets..." | tee logs_hgnn/main_$(date +%Y%m%d_%H%M%S).log
for dataset in "${coauthorship_datasets[@]}"; do
    run_dataset_experiments "coauthorship" "$dataset"
done

# Run experiments for cocitation datasets
echo "Running cocitation datasets..." | tee logs_hgnn/main_$(date +%Y%m%d_%H%M%S).log
for dataset in "${cocitation_datasets[@]}"; do
    run_dataset_experiments "cocitation" "$dataset"
done

echo "All experiments completed!" | tee logs_hgnn/main_$(date +%Y%m%d_%H%M%S).log
echo "Total end time: $(date)" | tee logs_hgnn/main_$(date +%Y%m%d_%H%M%S).log 