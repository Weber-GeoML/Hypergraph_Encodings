import os
import re
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import json


def parse_log_file(file_path):
    """Extract best test accuracy, std, and config from filename and content."""
    try:
        # Parse config from filename
        filename = os.path.basename(file_path)
        parts = filename.replace(".log", "").split("_")

        config_dict = {
            "model": parts[0],
            "data": parts[1],
            "dataset": parts[2],
        }

        # Add encoding info if present
        if len(parts) > 3:
            if parts[3] == "no_encodings":
                config_dict["add_encodings"] = False
            else:
                config_dict["add_encodings"] = True
                config_dict["encodings"] = parts[3]
                # Add additional encoding parameters if present
                current_idx = 4
                if len(parts) > current_idx:
                    if parts[3] == "RW":
                        config_dict["random_walk_type"] = parts[current_idx]
                        current_idx += 1
                    elif parts[3] == "LCP":
                        config_dict["curvature_type"] = parts[current_idx]
                        current_idx += 1
                    elif parts[3] == "Laplacian":
                        config_dict["laplacian_type"] = parts[current_idx]
                        current_idx += 1

                # Add transformer info if present
                if len(parts) > current_idx and "transformer" in parts[current_idx]:
                    config_dict["do_transformer"] = True
                    current_idx += 1  # Skip 'transformerTrue'
                    if len(parts) > current_idx:
                        config_dict["transformer_version"] = parts[current_idx]
                        current_idx += 1
                    if len(parts) > current_idx and "depth" in parts[current_idx]:
                        config_dict["transformer_depth"] = parts[current_idx].replace(
                            "depth", ""
                        )

        # Extract accuracy from last line
        with open(file_path, "r") as f:
            lines = f.readlines()
            if not lines:
                print(f"Empty file: {file_path}")
                return None, None, None

            last_line = lines[-1].strip()
            acc_match = re.search(
                r"Average best test accuracy: ([\d.]+) ± ([\d.]+)", last_line
            )

            if not acc_match:
                print(f"Could not find accuracy pattern in file: {file_path}")
                print("Last few lines of file:")
                print("\n".join(lines[-5:]))  # Print last 5 lines
                return None, None, None

            acc = float(acc_match.group(1))
            std = float(acc_match.group(2))

            return acc, std, config_dict

    except Exception as e:
        print(f"\nError processing file {file_path}:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return None, None, None


def create_results_table(log_dir):
    results = {}
    full_results = {
        "train_accs": {},
        "val_accs": {},
        "test_accs": {},
        "params": {},
        "config_args": {},
    }

    # Check if log directory exists
    if not os.path.exists(log_dir):
        print(f"Error: Log directory '{log_dir}' does not exist!")
        return None, None

    # Count log files
    log_files = [f for f in os.listdir(log_dir) if f.endswith(".log")]
    if not log_files:
        print(f"Error: No .log files found in '{log_dir}'")
        return None, None

    print(f"Found {len(log_files)} log files")

    # Process all log files
    processed_files = 0
    for filename in log_files:
        try:
            # Parse filename components
            parts = filename.replace(".log", "").split("_")
            model = parts[0]
            data_type = parts[1]
            dataset = parts[2]

            # Determine encoding type
            if len(parts) > 3:
                if parts[3] == "noencodings":
                    encoding = "No Encoding"
                else:
                    encoding = "_".join(parts[3:])
            else:
                encoding = "No Encoding"

            # Create key for dataset
            if data_type == "coauthorship":
                dataset_key = f"{dataset}-CA"
            elif data_type == "cocitation":
                dataset_key = f"{dataset}-CC"
            else:
                dataset_key = dataset

            # Parse results and config
            acc, std, config_args = parse_log_file(os.path.join(log_dir, filename))
            if acc is not None:
                key = (model, encoding)
                if key not in results:
                    results[key] = {}
                results[key][dataset_key] = (acc, std)

                # Store full configuration
                run_key = f"{model}_{data_type}_{dataset}_{encoding}"
                full_results["config_args"][run_key] = config_args
                full_results["params"][run_key] = {
                    "model": model,
                    "data_type": data_type,
                    "dataset": dataset,
                    "encoding": encoding,
                    "accuracy": acc,
                    "std": std,
                    **config_args,
                }
                processed_files += 1
            else:
                print(f"Warning: Could not extract results from {filename}")

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

    print(f"Successfully processed {processed_files} out of {len(log_files)} files")

    if not results:
        print("Error: No valid results were extracted from the log files")
        return None, None

    # Create DataFrame for the table
    datasets = sorted(list(set(d for r in results.values() for d in r.keys())))
    rows = []
    for (model, encoding), data in sorted(results.items()):
        row = []
        # Get transformer info from config if available
        run_key = next(iter(data.keys()))  # Get any dataset key to access config
        config = full_results["config_args"].get(
            f"{model}_{data_type}_{run_key}_{encoding}", {}
        )

        # Build model name with transformer info if present
        if config.get("do_transformer", False):
            transformer_info = (
                f" (T{config['transformer_version']}d{config['transformer_depth']})"
            )
        else:
            transformer_info = ""

        model_name = f"{model}{transformer_info}"
        if encoding != "No Encoding":
            model_name += f" ({encoding})"

        row.append(model_name)
        for dataset in datasets:
            if dataset in data:
                acc, std = data[dataset]
                row.append(f"{acc:.2f} ± {std:.2f}")
            else:
                row.append("")
        rows.append(row)

    if not rows:
        print("Error: No rows generated for the table")
        return None, None

    df = pd.DataFrame(rows, columns=["Model"] + datasets)

    # Save results
    results_dir = os.path.join(log_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save LaTeX table
    latex_table = df.to_latex(index=False, escape=False)
    with open(os.path.join(results_dir, "results_table.tex"), "w") as f:
        f.write(latex_table)

    # Save full results
    with open(os.path.join(results_dir, "full_results.json"), "w") as f:
        json.dump(full_results, f, indent=4)

    # Create and save visual table
    if len(df) > 0:  # Only create plot if we have data
        fig, ax = plt.subplots(figsize=(15, len(rows) * 0.5))
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(
            cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
        )

        # Adjust table style
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        plt.title("Results Summary")
        plt.tight_layout()
        plt.savefig(
            os.path.join(results_dir, "results_table.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"\n=== Results Saved in {results_dir} ===")
        print(f"LaTeX table saved as: results_table.tex")
        print(f"Full results saved as: full_results.json")
        print(f"Summary plot saved as: results_table.png")

    return df, full_results


# Use the function
log_dir = "logs_loops_general_1"  # Update this to your log directory
results_df, full_results = create_results_table(log_dir)
if results_df is not None:
    print("\nDataFrame view of results:")
    print(results_df)
else:
    print("\nNo results to display")
