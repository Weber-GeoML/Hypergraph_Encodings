import multiprocessing as mp
import os
import pickle
import warnings

from encodings_hnns.save_lukas_encodings_base_class import EncodingsSaverBase

warnings.simplefilter("ignore")


class encodings_saver_lrgb(EncodingsSaverBase):
    """Handles encoding computation for LRGB datasets"""

    def __init__(self, data: str) -> None:
        super().__init__("graph_classification_datasets")
        self.data = data

    def process_split(
        self, split_data: list, dataset_name: str, split: str
    ) -> dict[str, dict[str, list]]:
        """Process one split of the dataset (train/val/test)"""
        results: list[mp.pool.AsyncResult] = []
        with mp.Pool() as pool:
            for count, hg in enumerate(split_data):
                result = pool.apply_async(
                    self._process_hypergraph,
                    (hg, dataset_name, count, split),
                )
                results.append(result)
            pool.close()
            pool.join()

        # Combine results
        combined_results: dict[str, list] = {
            "rw_EE": [],
            "rw_EN": [],
            "rw_WE": [],
            "lape_hodge": [],
            "lape_normalized": [],
            "orc": [],
            "frc": [],
            "ldp": [],
        }

        for result in results:
            encodings = result.get()
            combined_results["rw_EE"].extend(encodings[0])
            combined_results["rw_EN"].extend(encodings[1])
            combined_results["rw_WE"].extend(encodings[2])
            combined_results["lape_hodge"].extend(encodings[3])
            combined_results["lape_normalized"].extend(encodings[4])
            combined_results["orc"].extend(encodings[5])
            combined_results["frc"].extend(encodings[6])
            combined_results["ldp"].extend(encodings[7])

        return combined_results

    def compute_encodings(
        self, converted_datasets: tuple
    ) -> dict[str, tuple[list, list, list, list, list, list, list, list]]:
        """Returns a dataset specific function to compute the
        encodings on the data added by Lukas

        Returns:
            a function to compute the encodings
        """

        all_data, train_data, val_data, test_data = converted_datasets

        results: dict[str, dict[str, list]] = {
            "all": self.process_split(all_data, self.data, "all"),
            "train": self.process_split(train_data, self.data, "train"),
            "val": self.process_split(val_data, self.data, "val"),
            "test": self.process_split(test_data, self.data, "test"),
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
