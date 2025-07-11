"""File used to save the encodings.

Will just save in the same format Lukas provided to me.ie a list of dict
And now the "features" field of every dict will have been updated with the
features.
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

        # Process hypergraph classification datasets
        print("Processing hypergraph classification datasets...")
        DATA_TYPE = "hypergraph_classification_datasets"
        encodings_saver_instance = EncodingsSaver(DATA_TYPE)
        parsed_data = encodings_saver_instance.compute_encodings()

        # Process Cocitation/Coauthorship datasets
        print("Processing Cocitation/Coauthorship datasets...")
        cc_ca_saver = EncodingsSaverForCCCA()
        cc_ca_results = cc_ca_saver.compute_encodings()

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
