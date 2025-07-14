#!/bin/bash
#SBATCH --job-name=lukas_new      # Job name
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --time=160:00:00         # Time limit (hh:mm:ss)
#SBATCH --mem=48GB               # Memory required
#SBATCH --output=lukas_file_new.log  # Standard output and error log (with job ID)
#SBATCH --partition=mweber_gpu     # Specify the partition
#SBATCH --gpus=4                   # Request 1 GPU

# Create a timestamp for the log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="encoding_computation_${TIMESTAMP}.log"

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
log_message "Starting encoding computation job"

# Load Conda (if needed)
module load anaconda/2023.07  # Example, depending on your system
export PATH="/n/home04/rpellegrinext/Hypergraph_Encodings/julia-1.9.3/bin:$PATH"  # If installed locally

# Activate the Conda environment
source activate hgencodings_gpu_weber
log_message "Conda environment activated: hgencodings_gpu_weber"

# Run the first script and log its output
# log_message "Running: python scripts/compute_encodings/compute_and_save_encodings.py"
# python scripts/compute_encodings/compute_and_save_encodings.py 2>&1 | tee -a "$LOG_FILE"

# Run the second script and log its output
log_message "Running: python scripts/compute_encodings/compute_and_save_encodings_cc_ca.py"
python scripts/compute_encodings/compute_and_save_encodings_cc_ca.py 2>&1 | tee -a "$LOG_FILE"

log_message "Job completed successfully"



                                                    
