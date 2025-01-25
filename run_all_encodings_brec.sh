#!/bin/bash
#SBATCH --job-name=brec_parallel       
#SBATCH --array=0-48%4           # 7 encodings * 7 categories = 49 combinations, max 4 concurrent
#SBATCH --time=24:00:00         
#SBATCH --mem=16GB               
#SBATCH --output=sbatch_logs/brec_parallel_%A_%a.log  # %A is job ID, %a is array index

# Load modules and activate environment (adjust as needed)
module load anaconda/2023.07
source activate your_env_name

# Define ranges
ENCODING_START=0
ENCODING_END=6
CATEGORY_START=0
CATEGORY_END=6

# Calculate total combinations for array index mapping
TOTAL_CATEGORIES=$((CATEGORY_END - CATEGORY_START + 1))

# Calculate encoding and category from array task ID
encoding=$((SLURM_ARRAY_TASK_ID / TOTAL_CATEGORIES))
category=$((SLURM_ARRAY_TASK_ID % TOTAL_CATEGORIES))

# Create log directory
log_dir="logs_brec"
mkdir -p "$log_dir"

# Construct log filename
log_file="$log_dir/brec_encoding${encoding}_category${category}.log"

echo "Running analysis with encoding: $encoding, category: $category"
python scripts/analyse_brec.py -e "$encoding" -c "$category" > "$log_file" 2>&1

echo "Analysis complete!" 