#!/bin/bash

add_encodings=(True False)
models=("UniGCN")
data_types=("cocitation" "coauthorship")
coauthorship_datasets=("cora" "dblp")
cocitation_datasets=("cora" "citeseer" "pubmed")
encodings=("LDP" "LCP" "Laplacian" "RW")
rw_types=("EE" "EN")
curvature_types=("FRC")
laplacian_types=("Hodge" "Normalized")

# Create a directory to store logs
log_dir="logs_loops_general_1"
mkdir -p "$log_dir"

# Loop over all combinations of models, data types, datasets, and encodings
for model in "${models[@]}"; do
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
                                log_file="$log_dir/${model}_${data_type}_${dataset}_${encoding}_${rw_type}.log"
                                echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding --random-walk-type=$rw_type"
                                python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding" --random-walk-type="$rw_type" > "$log_file" 2>&1
                            done
                        elif [ "$encoding" == "LCP" ]; then
                            for curvature_type in "${curvature_types[@]}"; do
                                log_file="$log_dir/${model}_${data_type}_${dataset}_${encoding}_${curvature_type}.log"
                                echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding --curvature-type=$curvature_type"
                                python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding" --curvature-type="$curvature_type" > "$log_file" 2>&1
                            done
                        elif [ "$encoding" == "Laplacian" ]; then
                            for laplacian_type in "${laplacian_types[@]}"; do
                                log_file="$log_dir/${model}_${data_type}_${dataset}_${encoding}_${laplacian_type}.log"
                                echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding --laplacian-type=$laplacian_type"
                                python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding" --laplacian-type="$laplacian_type" > "$log_file" 2>&1
                            done
                        else
                            log_file="$log_dir/${model}_${data_type}_${dataset}_${encoding}.log"
                            echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --encodings=$encoding"
                            python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --encodings="$encoding" > "$log_file" 2>&1
                        fi
                    done
                else
                    echo "Not adding encodings"
                    log_file="$log_dir/${model}_${data_type}_${dataset}_noencodings.log"
                    echo "Running: python scripts/train_val.py --add-self-loop --model=$model --data=$data_type --dataset=$dataset --add-encodings=False"
                    python scripts/train_val.py --add-self-loop --model="$model" --data="$data_type" --dataset="$dataset" --add-encodings=False > "$log_file" 2>&1
                fi
            done
        done
    done
done
