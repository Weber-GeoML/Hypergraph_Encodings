import os

from encodings_hnns.save_lukas_encodings_simplified import process_lrgb_dataset


def main():
    # Set up paths
    base_path = "/n/holyscratch01/mweber_lab/lrgb_datasets"
    if os.path.exists(
        "/Users/pellegrinraphael/Desktop/Repos_GNN/Hypergraph_Encodings/data"
    ):
        base_path = (
            "/Users/pellegrinraphael/Desktop/Repos_GNN/Hypergraph_Encodings/data"
        )

    # List of datasets to process
    datasets = [
        "peptidesstruct",
    ]

    # Process each dataset
    for dataset_name in datasets:
        try:
            process_lrgb_dataset(dataset_name, base_path)
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")


if __name__ == "__main__":
    main()
