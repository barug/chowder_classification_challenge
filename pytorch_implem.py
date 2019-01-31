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
from sklearn import metrics

from load import WSIFilesHandler
from pytorch_model import PytorchChowder, ChowderEnsembler

#torch.set_printoptions(precision=10, threshold=100000)

MAX_TILES_NBR = 1000
BATCH_SIZE = 10
EPOCHS = 30
LEARNING_RATE = 0.001
R = 10

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=Path,
                    help="Data containing the lupus slides")
parser.add_argument("--conf", required=True, type=Path,
                    help="Path of configuration file")

# parser.add_argument("--num_runs", required=True, type=int,
#                     help="Number of runs for the cross validation")
# parser.add_argument("--num_splits", default=5, type=int,
#                     help="Number of splits for the cross validation")


import yaml

if __name__ == "__main__":

    args = parser.parse_args()
    file_handler = WSIFilesHandler(args.data_dir, MAX_TILES_NBR)
    features_train, features_test, labels_train, labels_test = file_handler.load_files()

    # unique, counts = np.unique(labels_train, return_counts=True)
    # print(unique, counts, counts[1] / (counts[0] + counts[1]))
    # pos_ratio = torch.Tensor((counts[0] / counts[1],))
    # print('pos_ratio: ' + str(pos_ratio))
    
    # -------------------------------------------------------------------------
    # Pytorch implem

    with args.conf.open() as conf_file:
        conf = yaml.load(conf_file)
    print(yaml.dump(conf))
    

    ensemble = ChowderEnsembler(**conf)
    ensemble.load_dataset(features_train, features_test, labels_train, labels_test)    
    ensemble.train_models()
    preds_test = ensemble.compute_predictions(features_test)

    from pytorch_metrics import pred_metrics
    print(preds_test)
    preds_test[preds_test > 0.5] = 1
    preds_test[preds_test <= 0.5] = 0
    print(preds_test)
    pred_metrics(labels_test, preds_test)
    exit()

    file_handler.save_test_predictions(preds_test)

    #labels_test = labels_test.squeeze().astype(int)

    # EPOCHS = 1
    # model_name = '{}_B{}_EP{}_LR{}_R{}.pt'.format(args.data_dir, BATCH_SIZE, EPOCHS, LEARNING_RATE, R)
    # pt_chowder.save_model('models/{}'.format(model_name))
