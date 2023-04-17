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

from torch.utils.data import ConcatDataset, Subset
from models import myLinearModel
import torch.utils.tensorboard as tb

if __name__ == '__main__':
    # dataset = torch.load(f"embeddingDataset\\train\\512-chex-not-normalize\\embeddings_dataset_final.pt")
    dataset = torch.load(f"embeddingDataset\\train\\512-chex-not-normalize-frontal\\embeddings_dataset_final.pt")

    print(len(dataset))

    dataset_part1 = Subset(dataset, range(0, 175000))
    dataset_part2 = Subset(dataset, range(175000, len(dataset)))

    print(len(dataset_part1))
    print(len(dataset_part2))
    torch.save(dataset_part1, f"embeddingDataset\\train\\512-chex-not-normalize-frontal\\first175k.pt")
    torch.save(dataset_part2, f"embeddingDataset\\train\\512-chex-not-normalize-frontal\\last16027.pt")