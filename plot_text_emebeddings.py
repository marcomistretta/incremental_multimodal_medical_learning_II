'''
Questo può essere diviso in quanti task mi pare...
Il numero di task == numero di "epoche"
Split senza intersezione e lasciando val e test invariati
Qua posso provare tutte le loss che voglio

'''
import copy

import numpy
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_curve
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from HeatMapPlotter import heatmap, annotate_heatmap
from Trainer import Trainer
from health_multimodal.text.utils import get_cxr_bert_inference
from models import myLinearModel

seed_value = 27
torch.manual_seed(seed_value)
import random

random.seed(seed_value)
np.random.seed(seed_value)
# xxx

from DataRetrieval import create_prompts, basic_create_prompts


def get_pos_neg_text_emb(label_name):
    pos_prompt = prompts[label_name]["positive"]
    neg_prompt = prompts[label_name]["negative"]

    pos_prompt_embedding = bert.get_embeddings_from_prompt(pos_prompt, normalize=False)
    assert pos_prompt_embedding.shape[0] == len(pos_prompt)
    # if multiple_prompts:
    pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
    pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(device)

    neg_prompt_embedding = bert.get_embeddings_from_prompt(neg_prompt, normalize=False)
    assert neg_prompt_embedding.shape[0] == len(neg_prompt)
    # if multiple_prompts:
    neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
    neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(device)

    return pos_prompt_embedding, neg_prompt_embedding


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on:", device)

    class_list = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    '''
        Atelectasis: ATEL
        Cardiomegaly: CMG
        Consolidation: CONS
        Edema: EDE
        Pleural Effusion: PLEF
    '''
    abbrevviations = ["ATEL-pos", "ATEL-neg", "CMG-pos", "CMG-neg", "CONS-pos", "CONS-neg",
                      "EDE-pos", "EDE-neg", "PLEF-pos", "PLEF-neg"]

    bert = get_cxr_bert_inference()
    multiple_prompts = True
    if multiple_prompts:
        prompts = create_prompts(class_list)
    else:
        prompts = basic_create_prompts(class_list)

    embeddings = []
    # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
    # colors = ['r', 'r', 'g', 'g', 'b', 'b', 'c', 'c', 'm', 'm']
    # shapes = ['o', 's', 'v', '^', '*', '+', 'x', 'D', 'p', 'h']
    shapes = ['o', 'v', 'o', 'v', 'o', 'v', 'o', 'v', 'o', 'v']
    class_groups = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}
    group_colors = ['r', 'g', 'b', 'c', 'm']
    colors = [group_colors[class_groups[i]] for i in range(10)]

    for label_name in class_list:
        pos_emb, neg_emb = get_pos_neg_text_emb(label_name)
        embeddings.append(pos_emb)
        embeddings.append(neg_emb)

    embeddings = torch.stack(embeddings).cpu()

    # xxx perform PCA on the embeddings to reduce them to 2 dimensions
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    # plot the reduced embeddings
    for i in range(10):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], marker=shapes[i], c=colors[i], label=f'class{i}')
    plt.title("PCA multiple-prompts")
    legend_categories = {'r': 'ATEL', 'g': 'CMG', 'b': 'CONS', 'c': 'EDE', 'm': 'PLEF'}
    legend_shapes = {'o': 'Positive', 'v': 'Negative'}
    handles = []
    for color, category in legend_categories.items():
        handles.append(
            plt.Line2D([0], [0], marker='o', color='w', label=category, markerfacecolor=color, markersize=10))
    for shape, label in legend_shapes.items():
        handles.append(plt.Line2D([0], [0], marker=shape, color='w', label=label, markerfacecolor='k', markersize=10))
    plt.legend(handles=handles)
    plt.show()


    # xxx perform t-SNE on the embeddings to reduce them to 2 dimensions
    tsne = TSNE(n_components=2, metric="euclidean")
    reduced_embeddings = tsne.fit_transform(embeddings)
    # plot the reduced embeddings
    for i in range(10):
        plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], marker=shapes[i], c=colors[i], label=f'class{i}')
    plt.title("t-SNE multiple-prompts")
    handles = []
    for color, category in legend_categories.items():
        handles.append(
            plt.Line2D([0], [0], marker='o', color='w', label=category, markerfacecolor=color, markersize=10))
    for shape, label in legend_shapes.items():
        handles.append(plt.Line2D([0], [0], marker=shape, color='w', label=label, markerfacecolor='k', markersize=10))
    plt.legend(handles=handles)
    plt.show()