import multiprocessing as mp
import os
import pickle
import warnings

from encodings_hnns.save_lukas_encodings_base_class import EncodingsSaverBase

warnings.simplefilter("ignore")

# Define mapping of encoding types to their keys
ENCODING_MAP: dict[str, dict[str, str]] = {
    "Laplacian": {"Hodge": "lape_hodge", "Normalized": "lape_normalized"},
    "LCP": {"ORC": "orc", "FRC": "frc"},
    "LDP": {"LDP": "ldp"},
    "RW": {"EE": "rw_EE", "EN": "rw_EN", "WE": "rw_WE"},
}


class encodings_saver_lrgb(EncodingsSaverBase):
    """Handles encoding computation for LRGB datasets"""

    def __init__(self, data: str) -> None:
        super().__init__("graph_classification_datasets")
        self.data = data

    def process_split(
        self,
        split_data: list,
        dataset_name: str,
        split: str,
        encodings_to_compute: str,
        laplacian_type: str,
        random_walk_type: str,
        curvature_type: str,
        verbose: bool = False,
    ) -> dict[str, list]:
        """Process one split of the dataset (train/val/test)"""
        results: list[mp.pool.AsyncResult] = []
        with mp.Pool() as pool:
            for count, hg in enumerate(split_data):
                result = pool.apply_async(
                    self._process_hypergraph,
                    (
                        hg,
                        dataset_name,
                        count,
                        verbose,
                        encodings_to_compute,
                        laplacian_type,
                        random_walk_type,
                        curvature_type,
                    ),
                )
                results.append(result)
            pool.close()
            pool.join()

        # Get the specific encoding type and key
        encoding_subtype: str = {
            "Laplacian": laplacian_type,
            "LCP": curvature_type,
            "RW": random_walk_type,
            "LDP": "LDP",
        }[encodings_to_compute]

        encoding_key: str = ENCODING_MAP[encodings_to_compute][encoding_subtype]

        # Combine results
        combined_results: dict[str, list] = {encoding_key: []}
        for result in results:
            encodings = result.get()
            combined_results[encoding_key].extend(encodings)

        return combined_results

    def compute_encodings(
        self,
        converted_datasets: tuple,
        encodings_to_compute: str,
        laplacian_type: str,
        random_walk_type: str,
        curvature_type: str,
    ) -> dict[str, dict[str, list]]:
        """Returns a dataset specific function to compute the
        encodings on the data added by Lukas

        Returns:
            a function to compute the encodings
        """

        all_data, train_data, val_data, test_data = converted_datasets

        results: dict[str, dict[str, dict[str, list]]] = {
            "all": self.process_split(
                all_data,
                self.data,
                "all",
                encodings_to_compute,
                laplacian_type,
                random_walk_type,
                curvature_type,
            ),
            "train": self.process_split(
                train_data,
                self.data,
                "train",
                encodings_to_compute,
                laplacian_type,
                random_walk_type,
                curvature_type,
            ),
            "val": self.process_split(
                val_data,
                self.data,
                "val",
                encodings_to_compute,
                laplacian_type,
                random_walk_type,
                curvature_type,
            ),
            "test": self.process_split(
                test_data,
                self.data,
                "test",
                encodings_to_compute,
                laplacian_type,
                random_walk_type,
                curvature_type,
            ),
        }

        # Save combined results
        for split, split_results in results.items():
            for encoding_type, encoded_data in split_results.items():
                save_file_path: str = (
                    f"{self.data}_{split}_with_encodings_{encoding_type}.pickle"
                )
                with open(os.path.join(self.d, save_file_path), "wb") as handle:
                    pickle.dump(encoded_data, handle)
                    print(f"Saved {len(encoded_data)} graphs to {save_file_path}")

        return results
