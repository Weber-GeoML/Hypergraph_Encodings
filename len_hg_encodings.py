import os
import pickle

# Directory where the pickle files are stored
# directory = '/Users/pellegrinraphael/Desktop/Repos_GNN/Hypergraph_Encodings/data/hypergraph_classification_datasets'
directory = "/Users/pellegrinraphael/Desktop/Repos_GNN/Hypergraph_Encodings/data/hypergraph_classification_datasets"
# Iterate over all files in the directory

# Get all pickle files and sort them alphabetically
filenames = sorted([f for f in os.listdir(directory) if f.endswith(".pickle")])


for filename in filenames:
    if filename.endswith(".pickle"):
        file_path = os.path.join(directory, filename)

        # Load the pickle file
        with open(file_path, "rb") as f:
            data = pickle.load(f)

            # Check if the loaded data is a list
            if isinstance(data, list):
                print(f"{filename}: {len(data)} elements")
            else:
                print(f"{filename}: Not a list")
