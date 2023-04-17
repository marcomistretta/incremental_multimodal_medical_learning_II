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

from torch.utils.data import ConcatDataset
from models import myLinearModel
import torch.utils.tensorboard as tb
# xxx SET REPRODUCIBILITY
# import torch
# seed_value = 42
# # set Python random seed
# import random
# random.seed(seed_value)
# # set NumPy random seed
# import numpy as np
# np.random.seed(seed_value)
# # set PyTorch random seed
# torch.manual_seed(seed_value)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
seed_value = 27
torch.manual_seed(seed_value)
import random
if __name__ == '__main__':
    index_list = [str(i * 5000) for i in range(1, int(200000 / 5000))]
    print(index_list)

    dataset = ConcatDataset([torch.load(f"embeddingDataset\\train\\512-chex-not-normalize-frontal\\embeddings_dataset_{i}.pt") for i in index_list])

    torch.save(dataset, "embeddingDataset/train/512-chex-not-normalize-frontal/embeddings_dataset_final_old.pt")