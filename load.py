import argparse
from pathlib import Path

import numpy as np
import pandas as pd


class WSIDataset:

    def __init__(self, ids, features, meta, paddings, labels):
        self.ids = ids
        self.features = features
        self.meta = meta
        self.paddings = paddings
        self.labels = labels

class WSIDataHandler:
    
    def __init__(self, data_dir, max_tiles_nbr):
        assert data_dir.is_dir()
        self.data_dir = data_dir
        self.max_tiles_nbr = max_tiles_nbr

        train_dir = data_dir / "train_input" / "resnet_features"
        test_dir = data_dir / "test_input"  / "resnet_features"

        train_labels_filename = data_dir / "train_output.csv"
        test_labels_filename = data_dir / "test_output.csv"
        self.train_dataset = self._load_dataset(train_dir, train_labels_filename)
        self.test_dataset = self._load_dataset(test_dir, test_labels_filename)

    def _load_dataset(self, features_dir, labels_file):
        labels = pd.read_csv(labels_file)
        filenames = [features_dir / "{}.npy".format(idx) for idx in labels["ID"]]
        for filename in filenames:
            assert filename.is_file(), filename
        ids = [f.stem for f in filenames]
        features, meta, paddings = self._load_features(filenames)
        labels = labels["Target"].values
        return WSIDataset(ids, features, meta, paddings, labels)
        
    def _load_features(self, filenames):
        # Load numpy arrays
        features = []
        meta = []
        paddings = []
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
            paddings.append((left_pad, right_pad))
        
        features = np.stack(features, axis = 0)
        meta = np.stack(meta, axis = 0)
        paddings = np.stack(paddings, axis = 0)
        return features, meta, paddings

    def save_test_predictions(self, preds):
        # Write the predictions in a csv file, to export them to the data challenge platform
        test_output = pd.DataFrame({"ID": self.test_dataset.ids, "Target": preds})
        test_output.set_index("ID", inplace=True)
        test_output.to_csv(self.data_dir / "test_pytorch.csv")

    def aggregate_tiles_meta(self, scores):
        scores_t = scores.transpose((0, 2, 1))
        full_meta = np.append(self.test_dataset.meta, scores_t, axis=2)
        meta_dict = {}
        for i in range(full_meta.shape[0]):
            patient_id = self.test_dataset.ids[i]
            left_pad, right_pad = self.test_dataset.paddings[i]
            p_res = full_meta[i, left_pad : 1000 - right_pad]
            meta_dict[patient_id] = p_res
        return meta_dict



