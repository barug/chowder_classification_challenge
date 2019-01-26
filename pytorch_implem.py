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

from sklearn.metrics import confusion_matrix

MAX_TILES_NBR = 1000
BATCH_SIZE = 10
EPOCHS = 30
LEARNING_RATE = 0.001
R = 5

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
        patient_features = patient_features[:, 3:]
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(suppress=True)
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
        #print(res)
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

    # unique, counts = np.unique(labels_train, return_counts=True)
    # print(unique, counts, counts[1] / (counts[0] + counts[1]))
    # pos_ratio = torch.Tensor((counts[0] / counts[1],))
    # print('pos_ratio: ' + str(pos_ratio))
    
    # -------------------------------------------------------------------------
    # Pytorch implem

    train_dataset = TensorDataset(features_train, labels_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 

    valid_dataset = TensorDataset(features_test, labels_test)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE * 2, shuffle=True)

    model = nn.Sequential(
        nn.Conv1d(2048, 1, 1,bias=False),
        nn.Dropout(p=0.5),
        MinMax(R),
        nn.Linear(2 * R, 200,bias=False),
        nn.Sigmoid(),
        nn.Dropout(p=0.5),
        nn.Linear(200, 100,bias=False),
        nn.Sigmoid(),
        nn.Dropout(p=0.5),
        nn.Linear(100, 1,bias=False)
        #nn.Sigmoid()
    ).float()

    def init_layers(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            #nn.init.constant_(m.bias.data, 0)
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data)
            #nn.init.constant_(m.bias.data, 0)

    model.apply(init_layers)

    torch.set_printoptions(precision=10)

    
    #loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_ratio)
    loss_func = nn.BCEWithLogitsLoss()
    
    params_conv = list(model[0].parameters())
    params_others = [param for layer in model[1:] for param in layer.parameters()]
    opt = optim.Adam([
            {'params': params_conv, 'weight_decay': 0.5},
            {'params': params_others}
        ], lr=LEARNING_RATE)
    #opt = optim.Adam(model.parameters(), LEARNING_RATE)

    # print(list(model[2].parameters()))
    # exit()
    for i in range(EPOCHS):
        model.train()
        for xb, yb in train_dataloader:
            opt.zero_grad()
            y_pred = model(xb)
            # print(yb)
            # print(y_pred)
            loss = loss_func(y_pred, yb)
            loss.backward()

            # for param in model[2].parameters():
            #     print(str(param.size()))
            #     print()
            #     print(param.grad)
            #     exit()
            opt.step()
            
        

        model.eval()

        remove_handles = []

        def apply_hook(m):
            def forward_metrics(module, input, output):
                print(type(module).__name__)
                print(output)
            if isinstance(m, (nn.Conv1d, nn.Linear, nn.Sigmoid)):
                remove_handles.append(m.register_forward_hook(forward_metrics))

        model.apply(apply_hook)
        model(xb)

        for rm_hook in remove_handles:
            rm_hook.remove()

        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dataloader)
        
        print(i, valid_loss, valid_loss / len(valid_dataset))

    with torch.no_grad():
        preds_test = nn.Sigmoid().forward(model(features_test).squeeze()).numpy()
        print(preds_test)
        preds_test[preds_test > 0.5] = 1
        preds_test[preds_test <= 0.5] = 0
    
    print(preds_test)

    preds_test = preds_test.astype(int)
    labels_test = labels_test.squeeze().numpy().astype(int)
    print(preds_test)
    print(labels_test)
    # Check that predictions are in [0, 1]
    assert np.max(preds_test) <= 1.0
    assert np.min(preds_test) >= 0.0

    model_name = '{}_B{}_EP{}_LR{}_R{}.pt'.format(args.data_dir, BATCH_SIZE, EPOCHS, LEARNING_RATE, R)
    
    torch.save(model, 'models/{}'.format(model_name))

    print(confusion_matrix(labels_test, preds_test))


    # -------------------------------------------------------------------------
    # Write the predictions in a csv file, to export them to the data challenge platform
    test_output = pd.DataFrame({"ID": ids_test, "Target": preds_test})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(args.data_dir / "test_pytorch.csv")
