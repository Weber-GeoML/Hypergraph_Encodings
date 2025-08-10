#!/bin/bash
#SBATCH --job-name=hgnn_experiments     # Job name
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --time=24:00:00                # Time limit (hh:mm:ss) - 7 days
#SBATCH --mem=32GB                      # Memory required
#SBATCH --output=/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/logs_hgnn_exps/hgnn_citeseer_%j.log  # Standard output and error log (with job ID)
#SBATCH --partition=mweber_gpu          # Specify the partition
#SBATCH --gpus=1                        # Request 1 GPU

# Weights & Biases configuration
export WANDB_API_KEY="ea7c6eeb5a095b531ef60cc784bfeb87d47ea0b0"
export WANDB_ENTITY="weber-geoml-harvard-university"
export WANDB_PROJECT="hgnn-experiments-2"

# Activate the Conda environment
source activate hgencodings_gpu_weber

# Create logs directory in lab space
mkdir -p /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/logs_hgnn_exps

# Function to run experiments for a single dataset
run_dataset_experiments() {
    local data_type=$1
    local dataset_name=$2
    local log_file="/n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/logs_hgnn_exps/${data_type}_${dataset_name}_hgnn_$(date +%Y%m%d_%H%M%S).log"
    
    echo "============================================" | tee -a "$log_file"
    echo "Starting HGNN experiments for ${data_type}/${dataset_name}" | tee -a "$log_file"
    echo "Log file: $log_file" | tee -a "$log_file"
    echo "============================================" | tee -a "$log_file"
    echo "Start time: $(date)" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
    
    # Run the HGNN script with best hyperparameters and wandb logging
    python hgnn_m3.py \
        --data "$data_type" \
        --dataset "$dataset_name" \
        --n_runs 80 \
        --use_best_params \
        --wandb_enabled \
        --wandb_project "hgnn-experiments-2" \
        --wandb_entity "weber-geoml-harvard-university" 2>&1 | tee -a "$log_file"
    
    echo "Completed experiments for ${data_type}/${dataset_name}" | tee -a "$log_file"
    echo "End time: $(date)" | tee -a "$log_file"
    echo "============================================" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
}

# Main execution
echo "Starting HGNN UniGNN-Compatible Experiments" | tee /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/logs_hgnn_exps/main_$(date +%Y%m%d_%H%M%S).log
echo "Total start time: $(date)" | tee /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/logs_hgnn_exps/main_$(date +%Y%m%d_%H%M%S).log
echo "============================================" | tee /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/logs_hgnn_exps/main_$(date +%Y%m%d_%H%M%S).log


# Run experiments for cocitation datasets
echo "Running cocitation datasets..." | tee /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/logs_hgnn_exps/main_$(date +%Y%m%d_%H%M%S).log
for dataset in "citeseer"; do
    run_dataset_experiments "cocitation" "$dataset"
done

echo "All experiments completed!" | tee /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/logs_hgnn_exps/main_$(date +%Y%m%d_%H%M%S).log
echo "Total end time: $(date)" | tee /n/holylabs/LABS/mweber_lab/Everyone/rpellegrin/logs_hgnn_exps/main_$(date +%Y%m%d_%H%M%S).log 
