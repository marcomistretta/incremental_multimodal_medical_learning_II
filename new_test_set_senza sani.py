import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_curve
from torch import nn, optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.utils.data import ConcatDataset, TensorDataset
from models import Adapter
import torch.utils.tensorboard as tb

if __name__ == '__main__':

    dataset = torch.load("embeddingDataset\\test\\512-chex-not-normalize\\embeddings_dataset_final_old.pt")
    print(len(dataset))
    if True: # test or val
        X, Y = dataset[:]

        # create a boolean mask for Y
        mask = torch.sum(Y, dim=1) > 0

        # use the mask to select relevant rows of X and Y
        X_sub = X[mask]
        Y_sub = Y[mask]

        # create a new TensorDataset with the selected X and Y
        sub_dataset = TensorDataset(X_sub, Y_sub)
    else: # train
        sub_datasets = []

        for sub_dataset in dataset.datasets:
            X, Y = sub_dataset[:]

            # create a boolean mask for Y
            mask = torch.sum(Y, dim=1) > 0

            # use the mask to select relevant rows of X and Y
            X_sub = X[mask]
            Y_sub = Y[mask]

            # create a new TensorDataset with the selected X and Y
            sub_dataset_sub = TensorDataset(X_sub, Y_sub)
            sub_datasets.append(sub_dataset_sub)

        # create a new ConcatDataset from the selected sub_datasets
        sub_dataset = ConcatDataset(sub_datasets)

    torch.save(sub_dataset, "embeddingDataset\\test\\512-chex-not-normalize\\embeddings_dataset_final_new.pt")
    print(len(sub_dataset))

