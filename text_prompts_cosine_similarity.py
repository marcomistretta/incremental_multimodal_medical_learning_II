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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_curve
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from HeatMapPlotter import heatmap, annotate_heatmap
from Trainer import Trainer
from health_multimodal.text.utils import get_cxr_bert_inference
from models import myLinearModel

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

    tasks_order = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    bert = get_cxr_bert_inference()
    multiple_prompts = True
    if multiple_prompts:
        prompts = create_prompts(class_list)
    else:
        prompts = basic_create_prompts(class_list)

    cosine_similarity_heatmap = torch.zeros((10, 10))  # todo

    for i, label_name_i in enumerate(class_list):
        # print(i, label_name_i)
        pos_emb_i, neg_emb_i = get_pos_neg_text_emb(label_name_i)
        for j, label_name_j in enumerate(class_list):
            # print(j, label_name_j)
            pos_emb_j, neg_emb_j = get_pos_neg_text_emb(label_name_j)
            pos_similarities_left = torch.matmul(pos_emb_i, pos_emb_j)
            cosine_similarity_heatmap[i * 2, j * 2] = pos_similarities_left

            pos_similarities_right = torch.matmul(pos_emb_i, neg_emb_j)
            cosine_similarity_heatmap[i * 2, j * 2 + 1] = pos_similarities_right

            neg_similarities_left = torch.matmul(neg_emb_i, pos_emb_j)
            cosine_similarity_heatmap[i * 2 + 1, j * 2] = neg_similarities_left

            neg_similarities_right = torch.matmul(neg_emb_i, neg_emb_j)
            cosine_similarity_heatmap[i * 2 + 1, j * 2 + 1] = neg_similarities_right

    heat_map = numpy.array(cosine_similarity_heatmap)
    fig, ax = plt.subplots()
    im, cbar = heatmap(heat_map, [i for i in abbrevviations], [j for j in abbrevviations],
                       ax=ax,
                       cmap="YlGn", cbarlabel="Cosine similarity heatmap multiple-prompts")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    fig.tight_layout()
    plt.show()
