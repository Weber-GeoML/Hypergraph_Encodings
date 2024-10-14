#!/bin/bash
#SBATCH --job-name=ablationII       # Job name
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --time=168:00:00         # Time limit (hh:mm:ss)
#SBATCH --mem=16GB               # Memory required
#SBATCH --output=outputIIablation_%j.log  # Standard output and error log (with job ID)
#SBATCH --partition=mweber_gpu     # Specify the partition
#SBATCH --gpus=4                   # Request 1 GPU

# Load Conda (if needed)
module load anaconda/2023.07  # Example, depending on your system

# Activate the Conda environment
source activate hgencodings_gpu_weber

add_encodings=(False)
models=("UniGCNII")
data_types=("cocitation")
coauthorship_datasets=("cora" "dblp")
cocitation_datasets=("cora" "citeseer" "pubmed")
encodings=("RW" "LCP" "Laplacian" "LDP")
rw_types=("EE" "EN" "WE")
curvature_types=("FRC" "ORC")
laplacian_types=("Hodge" "Normalized")
nlayers=(16 32)

# Create a directory to store logs
log_dir="logs_loops_ablation"
mkdir -p "$log_dir"

# Loop over all combinations of models, data types, datasets, and encodings
for model in "${models[@]}"; do
    for numlayer in "${nlayers[@]}"; do
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
                                    log_file="$log_dir/${model}_${numlayer}_${data_type}_${dataset}_${encoding}_${rw_type}.log"
                                    echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding --random-walk-type=$rw_type  --nlayer=$numlayer"
                                    python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding" --random-walk-type="$rw_type" --nlayer="$numlayer" > "$log_file" 2>&1
                                done
                            elif [ "$encoding" == "LCP" ]; then
                                for curvature_type in "${curvature_types[@]}"; do
                                    log_file="$log_dir/${model}_${numlayer}_${data_type}_${dataset}_${encoding}_${curvature_type}.log"
                                    echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding --curvature-type=$curvature_type  --nlayer=$numlayer"
                                    python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding" --curvature-type="$curvature_type"  --nlayer="$numlayer"> "$log_file" 2>&1
                                done
                            elif [ "$encoding" == "Laplacian" ]; then
                                for laplacian_type in "${laplacian_types[@]}"; do
                                    log_file="$log_dir/${model}_${numlayer}_${data_type}_${dataset}_${encoding}_${laplacian_type}.log"
                                    echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding --laplacian-type=$laplacian_type  --nlayer=$numlayer"
                                    python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding" --laplacian-type="$laplacian_type"  --nlayer="$numlayer"> "$log_file" 2>&1
                                done
                            else
                                log_file="$log_dir/${model}_${numlayer}_${data_type}_${dataset}_${encoding}.log"
                                echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding  --nlayer=$numlayer"
                                python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding"  --nlayer="$numlayer" > "$log_file" 2>&1
                            fi
                        done
                    else
                        echo "Not adding encodings"
                        log_file="$log_dir/${model}_${numlayer}_${data_type}_${dataset}_noencodings.log"
                        echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --add-encodings=False"
                        python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --add-encodings=False --nlayer="$numlayer"> "$log_file" 2>&1
                    fi
                done
            done
        done
    done
done
