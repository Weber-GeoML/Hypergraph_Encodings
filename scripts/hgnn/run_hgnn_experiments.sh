#!/bin/bash
#SBATCH --job-name=hgnn_experiments     # Job name
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --time=24:00:00                # Time limit (hh:mm:ss) - 7 days
#SBATCH --mem=32GB                      # Memory required
#SBATCH --output=logs_hgnn_exps/hgnn_%j.log  # Standard output and error log (with job ID)
#SBATCH --partition=mweber_gpu          # Specify the partition
#SBATCH --gpus=1                        # Request 1 GPU

# Load required modules (adjust based on your cluster setup)
# module load anaconda/2023.07  # Example, depending on your system

# Activate the Conda environment

# Create logs directory
mkdir -p logs_hgnn_exps

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
    
    # Run the HGNN script with proper logging
    python hgnn_m3.py \
        --data "$data_type" \
        --dataset "$dataset_name" \
        --n_runs 80 \
        --epochs 500 \
        --patience 50 \
        --gpu 0 \
        --normalize_features \
        --normalize_encodings 2>&1 | tee -a "$log_file"
    
    echo "Completed experiments for ${data_type}/${dataset_name}" | tee -a "$log_file"
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
for dataset in "cora" "dblp"; do
    run_dataset_experiments "coauthorship" "$dataset"
done

# Run experiments for cocitation datasets
echo "Running cocitation datasets..." | tee logs_hgnn/main_$(date +%Y%m%d_%H%M%S).log
for dataset in "citeseer" "cora" "pubmed"; do
    run_dataset_experiments "cocitation" "$dataset"
done

echo "All experiments completed!" | tee logs_hgnn/main_$(date +%Y%m%d_%H%M%S).log
echo "Total end time: $(date)" | tee logs_hgnn/main_$(date +%Y%m%d_%H%M%S).log 
