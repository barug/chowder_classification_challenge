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

import pickle
import yaml

from load import WSIDataHandler
from pytorch_model import PytorchChowder, ChowderEnsembler
from my_metrics import pred_metrics

#torch.set_printoptions(precision=10, threshold=100000)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=Path,
                    help="Data containing the lupus slides")
parser.add_argument("--conf", required=True, type=Path,
                    help="Path of configuration file")


if __name__ == "__main__":

    args = parser.parse_args()
    
    with args.conf.open() as conf_file:
        conf = yaml.load(conf_file)    

    wsi_data = WSIDataHandler(args.data_dir, conf['mt'])
    
    ensemble = ChowderEnsembler(**conf)
    ensemble.load_dataset(wsi_data.train_dataset)    
    ensemble.train_models()
    preds_test = ensemble.compute_predictions(wsi_data.test_dataset.features)

    mtrcs = pred_metrics(wsi_data.test_dataset.labels, preds_test)

    scores = ensemble.compute_tiles_scores(wsi_data.test_dataset.features)
    patients_meta = wsi_data.aggregate_tiles_meta(scores)
    
    pickle.dump(patients_meta, open("results.p", "wb"))
    
    #np.save('results.npy', app_meta)

    
    
    #file_handler.save_test_predictions(preds_test)

    # model_name = '{}_B{}_EP{}_LR{}_R{}.pt'.format(args.data_dir, BATCH_SIZE, EPOCHS, LEARNING_RATE, R)
    # pt_chowder.save_model('models/{}'.format(model_name))
