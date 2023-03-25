import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from tqdm import tqdm


class Trainer:
    def __init__(self, model, bert, prompts, class_names, device, writer=None):
        self.model = model
        self.bert = bert
        self.prompts = prompts
        self.class_names = class_names
        self.device = device
        self.writer = writer

    def pre_test(self):
        pass
    def train(self, train_loader, optimizer, criterion, epoch, val_loader=None):
        self.model.train()
        batch_idx = -1
        for embs, labels in tqdm(train_loader, desc=f'Training epoch {epoch}', leave=False):
            batch_idx += 1
            embs, labels = embs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            new_embs = self.model(embs)
            new_embs = F.normalize(new_embs, dim=-1)

            predicted_labels = torch.zeros(labels.shape[0], 5).to(self.device)

            # Loop through each label
            i = 0
            for label_name in self.class_names:
                # Get the positive and negative prompts for the label
                pos_prompt = self.prompts[label_name]["positive"]
                neg_prompt = self.prompts[label_name]["negative"]

                # pos_prompt = pos_prompt.to(device)
                # neg_prompt = neg_prompt.to(device)
                # Encode the positive and negative prompts
                pos_prompt_embedding = self.bert.get_embeddings_from_prompt(pos_prompt, normalize=False)
                assert pos_prompt_embedding.shape[0] == len(pos_prompt)
                if len(pos_prompt) > 1:
                    pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
                pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(self.device)

                neg_prompt_embedding = self.bert.get_embeddings_from_prompt(neg_prompt, normalize=False)
                assert neg_prompt_embedding.shape[0] == len(neg_prompt)
                if len(neg_prompt) > 1:
                    neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
                neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(self.device)

                # Calculate the similarities between the image and the positive and negative prompts
                pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
                neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)

                # pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
                # neg_similarities = neg_similarities.reshape(-1, 1)
                # TODO UNICA GRANDISSIMA DIFFERENZA
                predicted_labels[:, i] = pos_similarities - neg_similarities  # XXX grandissima differnza
                i += 1

            loss = F.binary_cross_entropy_with_logits(predicted_labels, labels)
            loss.backward()
            optimizer.step()

            # Log training loss to TensorBoard
            if self.writer is not None:
                iteration = (epoch - 1) * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), iteration)

        if val_loader is not None:
            self.test(val_loader, criterion, epoch, is_val=True)

    def test(self, test_loader, criterion, epoch, is_val=False):
        self.model.eval()

        y_true = []
        y_pred = []
        if is_val:
            task = "Val"
        else:
            task = "Test"
        with torch.no_grad():
            count = 0
            for embs, labels in tqdm(test_loader, desc=task + " epoch " + str(epoch), leave=False):
                count += 1
                embs, labels = embs.to(self.device), labels.to(self.device)

                new_embs = self.model(embs)
                new_embs = F.normalize(new_embs, dim=-1)

                acc_predicted_labels = torch.zeros(labels.shape[0], 5).to(self.device)
                predicted_labels = torch.zeros(labels.shape[0], 5).to(self.device)

                # Loop through each label
                i = 0
                for label_name in self.class_names:
                    # Get the positive and negative prompts for the label
                    pos_prompt = self.prompts[label_name]["positive"]
                    neg_prompt = self.prompts[label_name]["negative"]

                    # pos_prompt = pos_prompt.to(device)
                    # neg_prompt = neg_prompt.to(device)
                    # Encode the positive and negative prompts
                    pos_prompt_embedding = self.bert.get_embeddings_from_prompt(pos_prompt, normalize=False)
                    assert pos_prompt_embedding.shape[0] == len(pos_prompt)
                    if len(pos_prompt) > 1:
                        pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
                    pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(self.device)

                    neg_prompt_embedding = self.bert.get_embeddings_from_prompt(neg_prompt, normalize=False)
                    assert neg_prompt_embedding.shape[0] == len(neg_prompt)
                    if len(neg_prompt) > 1:
                        neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
                    neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(self.device)

                    # Calculate the similarities between the image and the positive and negative prompts
                    pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
                    neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)

                    acc_pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
                    acc_neg_similarities = neg_similarities.reshape(-1, 1)

                    predicted_labels[:, i] = pos_similarities - neg_similarities  # XXX grandissima differnza

                    acc_predicted_labels[:, i] = torch.argmax(torch.cat([acc_neg_similarities, acc_pos_similarities], dim=1), dim=1) # XXX grandissima differnza
                    i += 1

                test_loss = F.binary_cross_entropy_with_logits(predicted_labels, labels)

                # Convert the predicted labels to a numpy array
                acc_predicted_labels = acc_predicted_labels.cpu().numpy()

                # Append the true and predicted labels to the lists
                y_true.append(labels.cpu().numpy())
                y_pred.append(acc_predicted_labels)

                if self.writer is not None:
                    self.writer.add_scalar(task + '/Loss', test_loss, (epoch-1)*len(test_loader)+count)

        # Concatenate the true and predicted labels
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # Calculate the metrics
        accuracy = accuracy_score(y_true, y_pred)
        # Log test loss and accuracy to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar(task + '/Accuracy', accuracy, epoch)

    def run(self, train_loader, test_loader, optimizer, criterion, epochs, val_loader=None):
        for epoch in range(1, epochs + 1):
            self.train(train_loader, optimizer, criterion, epoch, val_loader)
            self.test(test_loader, criterion, epoch)
