import argparse
from pathlib import Path

import numpy as np
import pandas as pd


class WSIDataset:

    def __init__(self, ids, features, meta, labels):
        self.ids = ids
        self.features = features
        self.meta = meta
        self.labels = labels

class WSIDataHandler:
    
    def __init__(self, data_dir, max_tiles_nbr):
        assert data_dir.is_dir()
        self.data_dir = data_dir
        self.max_tiles_nbr = max_tiles_nbr

        train_dir = self.data_dir / "train_input" / "resnet_features"
        test_dir = self.data_dir / "test_input"  / "resnet_features"

        train_labels_filename = self.data_dir / "train_output.csv"
        test_labels_filename = self.data_dir / "test_output.csv"
        
        self.train_dataset = self._load_dataset(train_dir, train_labels_filename)
        self.test_dataset = self._load_dataset(test_dir, test_labels_filename)

    def _load_dataset(self, features_dir, labels_file):
        labels = pd.read_csv(labels_file)
        filenames = [features_dir / "{}.npy".format(idx) for idx in labels["ID"]]
        for filename in filenames:
            assert filename.is_file(), filename
        ids = [f.stem for f in filenames]
        features, meta = self._load_features(filenames)
        labels = labels["Target"].values
        return WSIDataset(ids, features, meta, labels)
        
    def _load_features(self, filenames):
        # Load numpy arrays
        features = []
        meta = []
        for f in filenames:
            patient_features = np.load(f)

            patient_res = patient_features[:, 3:]
            patient_meta = patient_features[:, :3]
            pad_size = self.max_tiles_nbr - patient_res.shape[0]
            left_pad = pad_size // 2
            right_pad = pad_size // 2 + pad_size % 2
            padded_res = np.pad(patient_res, ((left_pad, right_pad), (0,0)), mode='constant', constant_values=(0,))
            padded_meta = np.pad(patient_meta, ((left_pad, right_pad), (0,0)), mode='constant', constant_values=(0,))
            padded_res = padded_res.transpose(1, 0)
            features.append(padded_res)
            meta.append(padded_meta)
        
        features = np.stack(features, axis = 0)
        meta = np.stack(meta, axis = 0)
        return features, meta


    # def save_test_predictions(self, preds):
    #     # Write the predictions in a csv file, to export them to the data challenge platform
    #     test_output = pd.DataFrame({"ID": self.ids_test, "Target": preds})
    #     test_output.set_index("ID", inplace=True)
    #     test_output.to_csv(self.data_dir / "test_pytorch.csv")


