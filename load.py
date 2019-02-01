import argparse
from pathlib import Path

import numpy as np
import pandas as pd


class WSIDataset:
    """Contains all information related to a WSI dataset"""
    
    def __init__(self, features_dir, labels_file, max_tiles_nbr):
        self.max_tiles_nbr = max_tiles_nbr
        labels = pd.read_csv(labels_file)
        filenames = [features_dir / "{}.npy".format(idx) for idx in labels["ID"]]
        for filename in filenames:
            assert filename.is_file(), filename
        ids = [f.stem for f in filenames]
        features, meta, paddings = self._load_features(filenames)
        labels = labels["Target"].values
        self.ids = ids
        self.features = features
        self.meta = meta
        self.paddings = paddings
        self. labels = labels
        
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

    def save_test_predictions(self, preds, dest):
        # Write the predictions in a csv file, to export them to the data challenge platform
        test_output = pd.DataFrame({"ID": self.ids, "Target": preds})
        test_output.set_index("ID", inplace=True)
        test_output.to_csv(dest)

    def aggregate_results(self, preds, tiles_scores):
        scores_t = tiles_scores.transpose((0, 2, 1))
        full_meta = np.append(self.meta, scores_t, axis=2)
        results = {}
        for i in range(full_meta.shape[0]):
            patient_id = self.ids[i]
            left_pad, right_pad = self.paddings[i]
            tiles_meta = full_meta[i, left_pad : 1000 - right_pad]
            results[patient_id] = {}
            results[patient_id]['prediction'] = preds[i]
            results[patient_id]['tiles'] = tiles_meta
        return results



