"""Train a baseline model, and make a prediction."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import torch 
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

MAX_TILES_NBR = 1000
BATCH_SIZE = 10
EPOCHS = 40
LEARNING_RATE = 0.001
R = 50

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=Path,
                    help="Data containing the lupus slides")
# parser.add_argument("--num_runs", required=True, type=int,
#                     help="Number of runs for the cross validation")
# parser.add_argument("--num_splits", default=5, type=int,
#                     help="Number of splits for the cross validation")


def load_features(filenames):
    # Load numpy arrays
    features = []
    for f in filenames:
        patient_features = np.load(f    )

        # Remove location features (but we could use them?)
        #patient_features = patient_features[:, 3:]
        
        pad_size = MAX_TILES_NBR - patient_features.shape[0]
        left_pad = pad_size // 2
        right_pad = pad_size // 2 + pad_size % 2
        padded_features = np.pad(patient_features, ((left_pad, right_pad), (0,0)), mode='constant', constant_values=(0,))

        padded_features = padded_features.transpose(1, 0)
        
        features.append(padded_features)

    features = np.stack(features, axis=0)
    return features


class MinMax(nn.Module):

    def __init__(self, R):
        super().__init__()
        self.R = R
    
    def forward(self, x):
        top, _ = torch.topk(x, R, sorted=True)
        bottom, _ = torch.topk(x, R, largest=False, sorted=True)
        res = torch.cat((top, bottom), dim=2)
        return res


if __name__ == "__main__":
    
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load the data
    assert args.data_dir.is_dir()

    train_dir = args.data_dir / "train_input" / "resnet_features"
    test_dir = args.data_dir / "test_input"  / "resnet_features"

    train_output_filename = args.data_dir / "train_output.csv"
    test_output_filename = args.data_dir / "test_output.csv"
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
    ids_test = [f.stem for f in filenames_test]

    features_train = torch.Tensor(load_features(filenames_train))
    features_test = torch.Tensor(load_features(filenames_test))

    # Get the labels
    labels_train = torch.Tensor(train_output["Target"].values).view(-1, 1, 1)
    assert len(filenames_train) == len(labels_train)
    labels_test = torch.Tensor(test_output["Target"].values).view(-1, 1, 1)
    assert len(filenames_test) == len(labels_test)
    
    # -------------------------------------------------------------------------
    # Pytorch implem

    train_dataset = TensorDataset(features_train, labels_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE) 

    valid_dataset = TensorDataset(features_test, labels_test)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE * 2)

    model = nn.Sequential(
        nn.Conv1d(2051, 1, 1),
        MinMax(R),
        nn.Linear(2 * R, 200),
        nn.Sigmoid(),
        nn.Linear(200, 100),
        nn.Sigmoid(),
        nn.Linear(100, 1),
        nn.Tanh()
    ).float()
    
    loss_func = nn.BCELoss().float()
    
    opt = optim.Adam(model.parameters(), LEARNING_RATE)

    for i in range(EPOCHS):
        for xb, yb in train_dataloader:
            y_pred = model(xb)
            loss = loss_func(y_pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()
        
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dataloader)
        
        print(i, valid_loss, valid_loss / len(valid_dataset))

    with torch.no_grad():
        preds_test = model(features_test).squeeze().numpy()

    # -------------------------------------------------------------------------
    # Write the predictions in a csv file, to export them to the data challenge platform
    test_output = pd.DataFrame({"ID": ids_test, "Target": preds_test})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(args.data_dir / "test_pytorch.csv")
