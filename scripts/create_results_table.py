"""
This script is used to create a results table from the log files in the log_dir directory.

It creates a latex table and a png table.
"""

import json
import os
import re
import textwrap

import matplotlib.pyplot as plt
import pandas as pd


def parse_log_file(file_path):
    """Extract best test accuracy, std, and config from filename and content."""
    try:
        # Parse config from filename
        filename = os.path.basename(file_path)
        parts = filename.replace(".log", "").split("_")

        config_dict = {
            "filename": filename,
            "model": parts[0],
            "data": parts[1],
            "dataset": parts[2],
        }

        # Add encoding info if present
        if len(parts) > 3:
            current_idx = 3
            if parts[current_idx] == "no_encodings":
                config_dict["add_encodings"] = False
                current_idx += 1
            else:
                config_dict["add_encodings"] = True
                config_dict["encodings"] = parts[current_idx]
                current_idx += 1
                # Add additional encoding parameters if present
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
                    config_dict["transformer_depth"] = parts[
                        current_idx
                    ].replace("depth", "")
                    current_idx += 1

            # Add nlayer info if present
            if len(parts) > current_idx and "nlayer" in parts[current_idx]:
                config_dict["nlayer"] = parts[current_idx].replace("nlayer", "")

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
                print("--------------------------------" * 2)
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
    failed_files = []  # List to store failed file names
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
            acc, std, config_args = parse_log_file(
                os.path.join(log_dir, filename)
            )
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
                failed_files.append(filename)  # Add to failed files list

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            failed_files.append(filename)  # Add to failed files list

    print(
        f"Successfully processed {processed_files} out of {len(log_files)} files"
    )
    if failed_files:
        print("Failed files:")
        for failed_file in failed_files:
            print(f" - {failed_file}")

    if not results:
        print("Error: No valid results were extracted from the log files")
        return None, None

    # Create DataFrame for the table
    datasets = sorted(list(set(d for r in results.values() for d in r.keys())))
    rows = []
    for (model, encoding), data in sorted(results.items()):
        row = []
        # Get transformer info from config if available
        run_key = next(
            iter(data.keys())
        )  # Get any dataset key to access config
        config = full_results["config_args"].get(
            f"{model}_{data_type}_{run_key}_{encoding}", {}
        )

        # Build model name with transformer info if present
        if config.get("do_transformer", False):
            transformer_info = f" (T_{config['transformer_version']}d_{config['transformer_depth']})"
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

    # Split tables by model
    models = (
        df["Model"].apply(lambda x: x.split()[0]).unique()
    )  # Get base model names
    results_dir = os.path.join(log_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save full results
    with open(os.path.join(results_dir, "full_results.json"), "w") as f:
        json.dump(full_results, f, indent=4)

    # Create separate tables for each model
    for model_name in models:
        model_df = df[df["Model"].str.startswith(model_name)]

        # Save LaTeX table for this model
        latex_table = model_df.to_latex(index=False, escape=False)
        with open(
            os.path.join(results_dir, f"results_table_{model_name}.tex"), "w"
        ) as f:
            f.write(latex_table)

        # Create and save visual table
        if len(model_df) > 0:
            # Function to wrap text
            def wrap_text(text, width=20):
                """Wrap text at specified width."""
                if isinstance(text, str):
                    return "\n".join(textwrap.wrap(text, width=width))
                return text

            # Apply text wrapping
            wrapped_df = model_df.copy()
            wrapped_df["Model"] = wrapped_df["Model"].apply(
                lambda x: wrap_text(x, width=25)
            )
            wrapped_df.columns = [
                wrap_text(col, width=15) for col in wrapped_df.columns
            ]

            # Calculate figure size
            n_rows, n_cols = len(wrapped_df) + 1, len(wrapped_df.columns)
            row_height = 1.5
            fig_height = min(n_rows * row_height, 60)  # Cap maximum height
            fig_width = n_cols * 2.5

            # Create figure and table
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.axis("tight")
            ax.axis("off")

            table = ax.table(
                cellText=wrapped_df.values,
                colLabels=wrapped_df.columns,
                cellLoc="center",
                loc="center",
                colWidths=[1.0 / n_cols] * n_cols,
            )

            # Adjust table style
            table.auto_set_font_size(False)
            table.set_fontsize(9)

            # Adjust cell heights
            for cell in table._cells.values():
                cell.set_height(row_height / n_rows)
                cell._text.set_horizontalalignment("center")
                cell._text.set_verticalalignment("center")
                cell._text.set_multialignment("center")

            plt.title(f"Results Summary - {model_name}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(results_dir, f"results_table_{model_name}.png"),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.5,
            )
            plt.close()

        print(f"\n=== Results for {model_name} Saved in {results_dir} ===")
        print(f"LaTeX table saved as: results_table_{model_name}.tex")
        print(f"Summary plot saved as: results_table_{model_name}.png")

    print("Full results saved as: full_results.json")
    return df, full_results


# Use the function
log_dir = "logs_loops_general_new"  # Update this to your log directory
results_df, full_results = create_results_table(log_dir)
if results_df is not None:
    print("\nDataFrame view of results:")
    print(results_df)
else:
    print("\nNo results to display")
