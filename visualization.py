import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, type=Path,
                    help="Data containing the lupus slides")
# parser.add_argument("--num_runs", required=True, type=int,
#                     help="Number of runs for the cross validation")
# parser.add_argument("--num_splits", default=5, type=int,
#                     help="Number of splits for the cross validation")


def load_concat_tiles(filenames):
    # Load numpy arrays
    features = np.empty((0,2048))
    for f in filenames:
        patient_features = np.load(f)

        #Remove location features (but we could use them?)
        patient_features = patient_features[:, 3:]        
        features = np.append(features, patient_features, axis=0)
    print(features.shape)
    print(features.min())
    print(features.max())
    return features


def pred_metrics(labels_testset, labels_pred):
    labels_testset = labels_testset.astype(int)
    print("confusion matrix: \n" + str(metrics.confusion_matrix(labels_testset, labels_pred)))
    print("accuracy : " + str(metrics.accuracy_score(labels_test, labels_pred)))




if __name__ == "__main__":
    
    args = parser.parse_args()

    train_dir = args.data_dir / "train_input" / "resnet_features"
    test_dir = args.data_dir / "test_input"  / "resnet_features"

    train_output_filename = args.data_dir / "train_output.csv"
    test_output_filename = args.data_dir / "test_output.csv"
    pytorch_filename = args.data_dir / "test_pytorch.csv"

    train_output = pd.read_csv(train_output_filename)

    # Get the labels
    labels_train = train_output["Target"].values
    # unique, counts = np.unique(labels_train, return_counts=True)
    # print(unique, counts, counts[1] / (counts[0] + counts[1]))

    test_output = pd.read_csv(test_output_filename)
    labels_test = test_output["Target"].values
    # unique, counts = np.unique(labels_test, return_counts=True)
    # print(unique, counts, counts[1] / (counts[0] + counts[1]))

    # test_pytorch = pd.read_csv(pytorch_filename)
    # test_pytorch = test_pytorch["Target"].values

    
    # Get the filenames for train
    filenames_train = [train_dir / "{}.npy".format(idx) for idx in train_output["ID"]]
    for filename in filenames_train:
        assert filename.is_file(), filename

    features_train = load_concat_tiles(filenames_train)

    # print(features_train)
    # print(np.max(features_train))
    # print(np.min(features_train))

    # n, bins, patches = plt.hist(features_train)
    # plt.plot(bins)
    # plt.show()
    
