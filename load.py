import argparse
from pathlib import Path

import numpy as np
import pandas as pd

class WSIFilesHandler:
    
    def __init__(self, data_dir, max_tiles_nbr):
        assert data_dir.is_dir()
        self.data_dir = data_dir
        self.max_tiles_nbr = max_tiles_nbr

    def load_features(self, filenames):
        # Load numpy arrays
        features = []
        for f in filenames:
            patient_features = np.load(f)

            # Remove location features (but we could use them?)
            patient_features = patient_features[:, 3:]
            np.set_printoptions(threshold=np.inf)
            np.set_printoptions(suppress=True)
            pad_size = self.max_tiles_nbr - patient_features.shape[0]
            left_pad = pad_size // 2
            right_pad = pad_size // 2 + pad_size % 2
            padded_features = np.pad(patient_features, ((left_pad, right_pad), (0,0)), mode='constant', constant_values=(0,))

            padded_features = padded_features.transpose(1, 0)
            features.append(padded_features)

        features = np.stack(features, axis=0)
        return features


    def load_files(self):
        train_dir = self.data_dir / "train_input" / "resnet_features"
        test_dir = self.data_dir / "test_input"  / "resnet_features"

        train_output_filename = self.data_dir / "train_output.csv"
        test_output_filename = self.data_dir / "test_output.csv"
        train_output = pd.read_csv(train_output_filename)
        test_output = pd.read_csv(test_output_filename)

        # Get the filenames for train
        filenames_train = [train_dir / "{}.npy".format(idx) for idx in train_output["ID"]]
        for filename in filenames_train:
            assert filename.is_file(), filename

        # Get the numpy filenames for test
        filenames_test = sorted(test_dir.glob("*.npy"))
        for filename in filenames_test:
            assert filename.is_file(), filename
        self.ids_test = [f.stem for f in filenames_test]

        features_train = self.load_features(filenames_train)
        features_test = self.load_features(filenames_test)

        # Get the labels
        labels_train = train_output["Target"].values
        labels_test = test_output["Target"].values

        return features_train, features_test, labels_train, labels_test

    def save_test_predictions(self, preds):
        # Write the predictions in a csv file, to export them to the data challenge platform
        test_output = pd.DataFrame({"ID": self.ids_test, "Target": preds})
        test_output.set_index("ID", inplace=True)
        test_output.to_csv(self.data_dir / "test_pytorch.csv")


