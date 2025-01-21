#!/bin/bash
#SBATCH --job-name=lukas_new      # Job name
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --time=160:00:00         # Time limit (hh:mm:ss)
#SBATCH --mem=48GB               # Memory required
#SBATCH --output=lukas_file_new.log  # Standard output and error log (with job ID)
#SBATCH --partition=mweber_gpu     # Specify the partition
#SBATCH --gpus=4                   # Request 1 GPU

# Load Conda (if needed)
module load anaconda/2023.07  # Example, depending on your system
export PATH="/n/home04/rpellegrinext/Hypergraph_Encodings/julia-1.9.3/bin:$PATH"  # If installed locally

# Activate the Conda environment
source activate hgencodings_gpu_weber

echo "Running: python src/encodings_hnns/save_lukas_encodings_simplified.py"
python src/encodings_hnns/save_lukas_encodings_simplified.py
                                                    
