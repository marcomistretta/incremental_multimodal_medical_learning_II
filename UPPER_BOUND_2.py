import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_curve
from torch import nn
import torch.nn.functional as F
from models import Adapter
from torch import nn, optim
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from DataRetrieval import DataRetrieval, basic_create_prompts, create_prompts
from health_multimodal.image import get_biovil_resnet
from health_multimodal.text.utils import get_cxr_bert, get_cxr_bert_inference
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_curve
from torch import nn, optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import ConcatDataset
from models import Adapter
from trainer import Trainer
import torch.utils.tensorboard as tb
from health_multimodal.text.utils import get_cxr_bert, get_cxr_bert_inference
from DataRetrieval import DataRetrieval, basic_create_prompts, create_prompts

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # todo swap with right device
    print("running on:", device)

    chex_competition = True
    batch_size = 16384  # 4096, 8192, 8192
    val_batch = 256  # 64, 128, 128
    lr = 30  # 0.0001, 1, 30
    epochs = 10
    basic_prompts = False

    if chex_competition:
        print("*** CHEX COMPETITION ***")
        chex_str = "-chex"
        class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    else:
        print("NO chex competition")
        class_names = ["Pleural Effusion", "Pneumothorax", "Atelectasis", "Pneumonia", "Consolidation"]
        chex_str = ""

    train_dataset = torch.load(
        "embeddingDataset\\train\\512" + chex_str + "-not-normalize\\embeddings_dataset_final.pt")
    val_dataset = torch.load("embeddingDataset\\val\\512" + chex_str + "-not-normalize\\embeddings_dataset_final.pt")
    test_dataset = torch.load("embeddingDataset\\test\\512" + chex_str + "-not-normalize\\embeddings_dataset_final.pt")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=None, batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4, pin_memory=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, sampler=None, batch_size=val_batch,
                                             shuffle=True,
                                             num_workers=4, pin_memory=True, drop_last=False) # 64
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, sampler=None, batch_size=val_batch,
                                              shuffle=True,
                                              num_workers=4, pin_memory=True, drop_last=False)

    # xxx notare che non c'è bisogno di una resnet
    model = Adapter().to(device)

    cxr_bert = get_cxr_bert_inference()
    if cxr_bert.is_in_eval():
        print("Bert is in eval mode")
    if basic_prompts:
        str_basic = "-NO-prompt"
        prompts = basic_create_prompts(class_names)
    else:
        str_basic = "-mean-prompt"
        prompts = create_prompts(class_names)

    # writer = SummaryWriter("./model/adapter-lr-" + str(lr) + "bs" + str(batch_size)+"-"+chex_str+str_basic+"-prompt")
    writer_path = "./fine-tuned/adapter-lr" + str(lr) + "-bs" + str(batch_size) + "-ep"+str(epochs) + chex_str + str_basic
    writer = SummaryWriter(writer_path)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pre_test = False
    if pre_test:
        model.eval()
        epoch = 0
        batch_idx = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for embs, labels in tqdm(test_loader, desc="Pretest on chexpert, Epoch " + str(epoch)):

                batch_idx += 1

                # image0_to_plt = embs[0]
                # image0_to_plt = image0_to_plt.permute(1, 2, 0)
                # # Plot the RGB tensor
                # plt.imshow(image0_to_plt)
                # plt.show()

                embs = embs.to(device)
                labels = labels.to(device)
                # new_embs = model(embs)
                new_embs = embs
                new_embs = F.normalize(new_embs, dim=-1)

                predicted_labels = torch.zeros(labels.shape[0], 5).to(device)

                # Loop through each label
                i = -1
                for label_name in class_names:
                    i += 1
                    # Get the positive and negative prompts for the label
                    pos_prompt = prompts[label_name]["positive"]
                    neg_prompt = prompts[label_name]["negative"]

                    # pos_prompt = pos_prompt.to(device)
                    # neg_prompt = neg_prompt.to(device)
                    # Encode the positive and negative prompts
                    pos_prompt_embedding = cxr_bert.get_embeddings_from_prompt(pos_prompt, normalize=False)
                    assert pos_prompt_embedding.shape[0] == len(pos_prompt)
                    if not basic_prompts:
                        pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
                    pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(device)

                    neg_prompt_embedding = cxr_bert.get_embeddings_from_prompt(neg_prompt, normalize=False)
                    assert neg_prompt_embedding.shape[0] == len(neg_prompt)
                    if not basic_prompts:
                        neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
                    neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(device)

                    # Calculate the similarities between the image and the positive and negative prompts
                    pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
                    neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)

                    pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
                    neg_similarities = neg_similarities.reshape(-1, 1)
                    # xxx NON E' DERIVABILE LOL Take the maximum similarity as the predicted label
                    predicted_labels[:, i] = torch.argmax(torch.cat([neg_similarities, pos_similarities], dim=1), dim=1)
                    # predicted_labels[:, i] = pos_similarities - neg_similarities  # XXX grandissima differnza

                # Convert the predicted labels to a numpy array
                predicted_labels_np = predicted_labels.cpu().numpy()

                # Append the true and predicted labels to the lists
                y_true.append(labels.cpu().numpy())
                y_pred.append(predicted_labels_np)

        # Concatenate the true and predicted labels
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # Calculate the metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        auroc = roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")

        # Calculate precision-recall curve for each class
        precision_curve = []
        recall_curve = []
        for i in range(5):
            precision_i, recall_i, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
            precision_curve.append(precision_i)
            recall_curve.append(recall_i)

        writer.add_scalar("Comparison Accuracy", accuracy, epoch)
        writer.add_scalar("Comparison F1 score", f1, epoch)
        writer.add_scalar("Comparison AUROC", auroc, epoch)
        for i in range(5):
            writer.add_scalar("Comparison Class Accuracy", accuracy_score(y_true[:, i], y_pred[:, i]), i)
            writer.add_scalar("Comparison Class Precision",
                              precision_score(y_true[:, i], y_pred[:, i], average="weighted"), i)
            writer.add_scalar("Comparison Class Recall",
                              recall_score(y_true[:, i], y_pred[:, i], average="weighted"), i)
        # for i in range(5):
        #     fig = plt.figure()
        #     plt.plot(recall_curve[i], precision_curve[i], label='Precision-Recall curve')
        #     plt.xlabel('Recall')
        #     plt.ylabel('Precision')
        #     plt.title('Precision-Recall Curve for Class ' + str(i))
        #     plt.legend(loc="lower left")
        #     writer.add_figure('Precision-Recall Curve for Class ' + str(i), fig)
    for epoch in range(1, epochs):
        batch_idx = 0
        # xxx ONE EPOCH TRAIN
        model.train()
        for embs, labels in tqdm(train_loader, desc="Fine-tuning on chexpert, Epoch " + str(epoch)):
            optimizer.zero_grad()

            batch_idx += 1

            # image0_to_plt = embs[0]
            # image0_to_plt = image0_to_plt.permute(1, 2, 0)
            # # Plot the RGB tensor
            # plt.imshow(image0_to_plt)
            # plt.show()

            embs = embs.to(device)
            labels = labels.to(device)
            new_embs = model(embs)
            new_embs = F.normalize(new_embs, dim=-1)

            predicted_labels = torch.zeros(labels.shape[0], 5).to(device)

            # Loop through each label
            i = -1
            for label_name in class_names:
                i += 1
                # Get the positive and negative prompts for the label
                pos_prompt = prompts[label_name]["positive"]
                neg_prompt = prompts[label_name]["negative"]

                # pos_prompt = pos_prompt.to(device)
                # neg_prompt = neg_prompt.to(device)
                # Encode the positive and negative prompts
                pos_prompt_embedding = cxr_bert.get_embeddings_from_prompt(pos_prompt, normalize=False)
                assert pos_prompt_embedding.shape[0] == len(pos_prompt)
                if not basic_prompts:
                    pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
                pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(device)

                neg_prompt_embedding = cxr_bert.get_embeddings_from_prompt(neg_prompt, normalize=False)
                assert neg_prompt_embedding.shape[0] == len(neg_prompt)
                if not basic_prompts:
                    neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
                neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(device)

                # Calculate the similarities between the image and the positive and negative prompts
                pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
                neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)

                # xxx pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
                # xxx neg_similarities = neg_similarities.reshape(-1, 1)
                # xxx NON E' DERIVABILE LOL Take the maximum similarity as the predicted label
                # xxx predicted_labels[:, i] = torch.argmax(torch.cat([neg_similarities, pos_similarities], dim=1), dim=1)
                predicted_labels[:, i] = pos_similarities - neg_similarities  # XXX grandissima differnza

            # Compute loss and backpropagate
            loss = criterion(predicted_labels, labels)
            loss.backward()
            optimizer.step()

            iteration = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), iteration)

        batch_idx = 0
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for embs, labels in tqdm(val_loader, desc="Validating on chexpert, Epoch " + str(epoch)):

                batch_idx += 1

                # image0_to_plt = embs[0]
                # image0_to_plt = image0_to_plt.permute(1, 2, 0)
                # # Plot the RGB tensor
                # plt.imshow(image0_to_plt)
                # plt.show()

                embs = embs.to(device)
                labels = labels.to(device)
                new_embs = model(embs)
                new_embs = F.normalize(new_embs, dim=-1)

                predicted_labels = torch.zeros(labels.shape[0], 5).to(device)

                # Loop through each label
                i = -1
                for label_name in class_names:
                    i += 1
                    # Get the positive and negative prompts for the label
                    pos_prompt = prompts[label_name]["positive"]
                    neg_prompt = prompts[label_name]["negative"]

                    # pos_prompt = pos_prompt.to(device)
                    # neg_prompt = neg_prompt.to(device)
                    # Encode the positive and negative prompts
                    pos_prompt_embedding = cxr_bert.get_embeddings_from_prompt(pos_prompt, normalize=False)
                    assert pos_prompt_embedding.shape[0] == len(pos_prompt)
                    if not basic_prompts:
                        pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
                    pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(device)

                    neg_prompt_embedding = cxr_bert.get_embeddings_from_prompt(neg_prompt, normalize=False)
                    assert neg_prompt_embedding.shape[0] == len(neg_prompt)
                    if not basic_prompts:
                        neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
                    neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(device)

                    # Calculate the similarities between the image and the positive and negative prompts
                    pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
                    neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)

                    pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
                    neg_similarities = neg_similarities.reshape(-1, 1)
                    # xxx NON E' DERIVABILE LOL Take the maximum similarity as the predicted label
                    predicted_labels[:, i] = torch.argmax(torch.cat([neg_similarities, pos_similarities], dim=1), dim=1)
                    # predicted_labels[:, i] = pos_similarities - neg_similarities  # XXX grandissima differnza

                # Convert the predicted labels to a numpy array
                predicted_labels_np = predicted_labels.cpu().numpy()

                # Append the true and predicted labels to the lists
                y_true.append(labels.cpu().numpy())
                y_pred.append(predicted_labels_np)

        # Concatenate the true and predicted labels
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # Calculate the metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        auroc = roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")

        # Calculate precision-recall curve for each class
        precision_curve = []
        recall_curve = []
        for i in range(5):
            precision_i, recall_i, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
            precision_curve.append(precision_i)
            recall_curve.append(recall_i)

        writer.add_scalar("Val Accuracy", accuracy, epoch)
        writer.add_scalar("Val Train F1 score", f1, epoch)
        writer.add_scalar("Val AUROC", auroc, epoch)
        for i in range(5):
            writer.add_scalar("Val Class Accuracy " + str(epoch), accuracy_score(y_true[:, i], y_pred[:, i]), i)
            writer.add_scalar("Val Class Precision " + str(epoch),
                              precision_score(y_true[:, i], y_pred[:, i], average="weighted"), i)
            writer.add_scalar("Val Class Recall " + str(epoch),
                              recall_score(y_true[:, i], y_pred[:, i], average="weighted"), i)
        # for i in range(5):
        #     fig = plt.figure()
        #     plt.plot(recall_curve[i], precision_curve[i], label='Precision-Recall curve')
        #     plt.xlabel('Recall')
        #     plt.ylabel('Precision')
        #     plt.title('Precision-Recall Curve for Class ' + str(i))
        #     plt.legend(loc="lower left")
        #     writer.add_figure('Precision-Recall Curve for Class ' + str(i), fig)

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for embs, labels in tqdm(test_loader, desc="Testing on chexpert"):

            batch_idx += 1

            # image0_to_plt = embs[0]
            # image0_to_plt = image0_to_plt.permute(1, 2, 0)
            # # Plot the RGB tensor
            # plt.imshow(image0_to_plt)
            # plt.show()

            embs = embs.to(device)
            labels = labels.to(device)
            new_embs = model(embs)
            new_embs = F.normalize(new_embs, dim=-1)

            predicted_labels = torch.zeros(labels.shape[0], 5).to(device)

            # Loop through each label
            i = -1
            for label_name in class_names:
                i += 1
                # Get the positive and negative prompts for the label
                pos_prompt = prompts[label_name]["positive"]
                neg_prompt = prompts[label_name]["negative"]

                # pos_prompt = pos_prompt.to(device)
                # neg_prompt = neg_prompt.to(device)
                # Encode the positive and negative prompts
                pos_prompt_embedding = cxr_bert.get_embeddings_from_prompt(pos_prompt, normalize=False)
                assert pos_prompt_embedding.shape[0] == len(pos_prompt)
                if not basic_prompts:
                    pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
                pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(device)

                neg_prompt_embedding = cxr_bert.get_embeddings_from_prompt(neg_prompt, normalize=False)
                assert neg_prompt_embedding.shape[0] == len(neg_prompt)
                if not basic_prompts:
                    neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
                neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(device)

                # Calculate the similarities between the image and the positive and negative prompts
                pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
                neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)

                pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
                neg_similarities = neg_similarities.reshape(-1, 1)
                # xxx NON E' DERIVABILE LOL Take the maximum similarity as the predicted label
                predicted_labels[:, i] = torch.argmax(torch.cat([neg_similarities, pos_similarities], dim=1), dim=1)
                # predicted_labels[:, i] = pos_similarities - neg_similarities  # XXX grandissima differnza

            # Convert the predicted labels to a numpy array
            predicted_labels_np = predicted_labels.cpu().numpy()

            # Append the true and predicted labels to the lists
            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted_labels_np)
    # Concatenate the true and predicted labels
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Calculate the metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    auroc = roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    # Calculate precision-recall curve for each class
    precision_curve = []
    recall_curve = []
    for i in range(5):
        precision_i, recall_i, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
        precision_curve.append(precision_i)
        recall_curve.append(recall_i)

    writer.add_scalar("Comparison Accuracy", accuracy, 0)
    writer.add_scalar("Comparison F1 score", f1, 0)
    writer.add_scalar("Comparison AUROC", auroc, 0)
    for i in range(5):
        writer.add_scalar("Comparison Class Recall",
                          recall_score(y_true[:, i], y_pred[:, i], average="weighted"), i)
        writer.add_scalar("Comparison Class Accuracy", accuracy_score(y_true[:, i], y_pred[:, i]), i)
        writer.add_scalar("Comparison Class Precision",
                          precision_score(y_true[:, i], y_pred[:, i], average="weighted"), i)
