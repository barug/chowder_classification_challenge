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
import pprint

from load import WSIDataset
from pytorch_model import PytorchChowder, ChowderEnsembler
from my_metrics import pred_metrics


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=Path,
                    help="Dataset location")
parser.add_argument("--conf", required=True, type=Path,
                    help="Path of configuration file")
parser.add_argument("--save_mod_dir", required=False, type=str,
                    help="save models to dir")
parser.add_argument("--save_res_to", required=False, type=str,
                    help="save results to dir")


if __name__ == "__main__":

    args = parser.parse_args()
    
    # load conf
    with args.conf.open() as conf_file:
        conf = yaml.load(conf_file)    

    # load train set
    train_dir = args.data_dir / "train_input" / "resnet_features"
    train_labels_filename = args.data_dir / "train_output.csv"
    train_dataset = WSIDataset(train_dir, train_labels_filename, conf['mt'])

    # load test set
    test_dir = args.data_dir / "test_input"  / "resnet_features"
    test_labels_filename = args.data_dir / "test_output.csv"
    test_dataset = WSIDataset(test_dir, test_labels_filename, conf['mt'])    

    # train ensemble
    ensemble = ChowderEnsembler(**conf)
    ensemble.train_models(train_dataset)
    
    # compute prediction and prediction metrics
    preds_test = ensemble.compute_predictions(test_dataset.features)
    mtrcs = pred_metrics(test_dataset.labels, preds_test)
    pprint.pprint(mtrcs)

    # compute tiles scores and aggregate the results 
    scores = ensemble.compute_tiles_scores(test_dataset.features)
    patients_results = test_dataset.aggregate_results(preds_test, scores)
    
    # save models and results
    if args.save_mod_dir is not None:
        ensemble.save_ensemble(args.save_mod_dir, args.data_dir)
    if args.save_res_to is not None:
        pickle.dump(patients_results, open(args.save_res_to, "wb"))
