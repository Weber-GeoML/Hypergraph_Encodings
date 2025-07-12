"""File used to save the encodings.

Saves the data in thesame format provided.

ie either a dict with the following fields:

- hypergraph
- features
- labels

or a list of dicts with the following fields:

- hypergraph
- features
- labels

And now the "features" field of every dict will have been updated with the
encodings.
"""

import os
import sys
import pickle
import warnings

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from compute_encodings.encoding_saver_hgraphs import EncodingsSaver
from compute_encodings.encoding_saver_lrgb import EncodingsSaverLRGB
from compute_encodings.encoding_saver_cc_ca import EncodingsSaverForCCCA

warnings.simplefilter("ignore")

# Run
if __name__ == "__main__":
    LOCAL = True
    if LOCAL:
        print("Running locally")

        # Actually CA and CC are now computed in
        # scripts/compute_encodings/compute_and_save_encodings_cc_ca.py
        # which means we do not need to compute them here and use
        # the encodings_saver_cc_ca.py file. STILL NEED TO CHECK
        # THEY PRODUCE THE SAME RESULTS!
        # # Process Cocitation/Coauthorship datasets
        # print("Processing Cocitation/Coauthorship datasets...")

        # # Process coauthorship datasets
        # print("Processing coauthorship datasets...")
        # coauthorship_saver = EncodingsSaverForCCCA("coauthorship")
        # coauthorship_results = coauthorship_saver.compute_encodings()

        # # Process cocitation datasets
        # print("Processing cocitation datasets...")
        # cocitation_saver = EncodingsSaverForCCCA("cocitation")
        # cocitation_results = cocitation_saver.compute_encodings()

        # Process hypergraph classification datasets
        print("Processing hypergraph classification datasets...")
        DATA_TYPE = "hypergraph_classification_datasets"
        encodings_saver_instance = EncodingsSaver(DATA_TYPE)
        parsed_data = encodings_saver_instance.compute_encodings()

    CLUSTER = False
    if CLUSTER:
        # Change this to your path
        BASE_PATH = "/n/holyscratch01/mweber_lab/lrgb_datasets"
        DATASETS = [
            "peptidesstruct",
        ]

        for dataset_name in DATASETS:
            print(f"Processing {dataset_name}...")
            encodings_saver_lrgb_instance = EncodingsSaverLRGB(DATA_TYPE)
            converted_dataset = (
                encodings_saver_lrgb_instance.load_and_convert_lrgb_datasets(
                    BASE_PATH, dataset_name
                )
            )

            # Save the converted dataset
            save_path = os.path.join("data", "graph_classification_datasets")
            os.makedirs(save_path, exist_ok=True)

            save_file = os.path.join(save_path, f"{dataset_name}_hypergraphs.pickle")
            with open(save_file, "wb") as handle:
                pickle.dump(converted_dataset, handle)
            print(f"Saved {len(converted_dataset)} graphs to {save_file}")
