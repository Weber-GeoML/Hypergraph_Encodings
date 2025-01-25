#!/bin/bash

# Array of encodings to process
encodings=(
    "LDP"
    "LCP-FRC"
    "RWPE"
    "LCP-ORC"
    "LAPE-Normalized"
    "LAPE-RW"
    "LAPE-Hodge"
)

# Maximum number of parallel processes
MAX_PARALLEL=4

# Counter for running processes
running=0

for encoding in "${encodings[@]}"; do
    # Run the Python script in the background
    python scripts/analyse_brec.py --encoding "$encoding" &
    
    # Increment counter
    ((running++))
    
    # If we've reached max parallel processes, wait for one to finish
    if ((running >= MAX_PARALLEL)); then
        wait -n
        ((running--))
    fi
done

# Wait for remaining processes to finish
wait

echo "All analyses complete!"

# Combine results
python scripts/combine_results.py 