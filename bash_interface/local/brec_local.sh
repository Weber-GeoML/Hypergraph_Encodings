#!/bin/bash

# Create log directory
log_dir="logs_brec_orc_local"
mkdir -p "$log_dir"

# Define ranges
ENCODING_START=0
ENCODING_END=6
CATEGORY_START=0
CATEGORY_END=6

# Total number of combinations to process
total_combinations=$(( (ENCODING_END - ENCODING_START + 1) * (CATEGORY_END - CATEGORY_START + 1) ))
current=0

# Loop through all combinations
for encoding in $(seq $ENCODING_START $ENCODING_END); do
    for category in $(seq $CATEGORY_START $CATEGORY_END); do
        current=$((current + 1))
        
        # Construct log filename
        log_file="$log_dir/brec_encoding${encoding}_category${category}.log"
        
        # Print progress
        echo "[$current/$total_combinations] Running analysis with encoding: $encoding, category: $category"
        
        # Run the analysis and redirect output to log file
        python scripts/brec/analyse_brec.py -e "$encoding" -c "$category" > "$log_file" 2>&1
        
        echo "✓ Analysis complete for encoding $encoding, category $category"
        echo "----------------------------------------"
    done
done

echo "All analyses complete!"