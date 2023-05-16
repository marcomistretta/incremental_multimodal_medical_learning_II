import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, precision_recall_curve
from torch import nn, optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import ConcatDataset
from models import myLinearModel
from trainer import Trainer
import torch.utils.tensorboard as tb
from health_multimodal.text.utils import get_cxr_bert, get_cxr_bert_inference
from DataRetrieval import DataRetrieval, basic_create_prompts, create_prompts


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # todo swap with right device
    print("running on:", device)

    chex_competition = True
    batch_size = 16384
    lr = 1000 # 0.00001
    epochs = 10
    basic_prompts = False

    if chex_competition:
        print("*** CHEX COMPETITION ***")
        chex_str = "chex-"
        class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    else:
        print("NO chex competition")
        # class_names = ["Pleural Effusion", "Pneumothorax", "Atelectasis", "Pneumonia", "Consolidation"]
        chex_str = ""

    train_dataset = torch.load("embeddingDataset\\train\\512-"+chex_str+"not-normalize\\embeddings_dataset_final.pt")
    val_dataset = torch.load("embeddingDataset\\val\\512-"+chex_str+"not-normalize\\embeddings_dataset_final.pt")
    test_dataset = torch.load("embeddingDataset\\test\\512-"+chex_str+"not-normalize\\embeddings_dataset_final.pt")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=None, batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4, pin_memory=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, sampler=None, batch_size=128,
                                                  shuffle=False,
                                                  num_workers=4, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, sampler=None, batch_size=128,
                                                  shuffle=False,
                                                  num_workers=4, pin_memory=True, drop_last=False)

    # xxx notare che non c'è bisogno di una resnet
    model = myLinearModel().to(device)

    cxr_bert = get_cxr_bert_inference()
    if cxr_bert.is_in_eval():
        print("Bert is in eval mode")
    if basic_prompts:
        str_basic = "NO"
        prompts = basic_create_prompts(class_names)
    else:
        str_basic = "mean-"
        prompts = create_prompts(class_names)

    # writer = SummaryWriter("./image_adapter/adapter-lr-" + str(lr) + "bs" + str(batch_size)+"-"+chex_str+str_basic+"-prompt")
    writer = SummaryWriter("./image_adapter/adapter-lr-" + str(lr) + "bs" + str(batch_size)+"-"+chex_str+str_basic+"prompt")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(model, cxr_bert, prompts, class_names, device, writer)
    trainer.run(train_loader, test_loader, optimizer, criterion, epochs,
                val_loader)  # xxx run will execute train and test
