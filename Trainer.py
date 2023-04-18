import copy
import math
from health_multimodal.text import get_cxr_bert_inference

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_curve
from torch.utils.data import Dataset, DataLoader, RandomSampler, ConcatDataset, TensorDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from tqdm import tqdm

from DataRetrieval import basic_create_prompts, create_prompts

# change_labels = False
# loss_name = "standard"  # "standard" "opzione2"
from HeatMapPlotter import heatmap, annotate_heatmap
from models import myLinearModel, myMLP

# zero shot o SHARED true, IMAGE true TEXT true
# oppure con SHARED false, IMAGE false TEXT false
SHARED = True  # True,  False # xxx shared ha precedenza (usare sempre shared true con image e text true)
# con shared false invece si può fare che ci pare
IMAGE_MODEL = True  # True, False
TEXT_MODEL = True  # True, False
MODEL_USED = "mlp"  # mlp, dense, "no-head"

CHANGE_LABELS = False

class Trainer:
    @staticmethod
    def compare_models(model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')

    @staticmethod
    def convert_1d_to_2d(arr):
        mask = torch.tensor(arr) == 1
        result = torch.zeros((len(arr), 2))
        result[mask, 1] = 1
        result[~mask, 0] = 1
        return result.tolist()

    def __init__(self, single_prompt, prompts, class_names, loss_name, lr, device, writer):
        self.bert_encoder = get_cxr_bert_inference()
        self.prompts = prompts
        self.class_names = class_names
        self.device = device
        self.writer = writer
        self.loss_name = loss_name
        self.change_labels = CHANGE_LABELS
        if self.change_labels:
            print("*** Watch out! Changing labels is enabled! ***")
        self.basic_prompts = single_prompt
        if single_prompt:
            print("Single prompt per class")
        else:
            print("Multiple prompts per class")
        print("*** LOSS " + str(loss_name) + " ***")
        params = []
        if SHARED:
            print("*** SHARED MODEL !!!! ***")
            if MODEL_USED == "mlp":
                shared_model = myMLP().to(device)
            elif MODEL_USED == "dense":
                shared_model = myLinearModel().to(device)
            else:
                print("*** ERROR... ***")
            global IMAGE_MODEL
            IMAGE_MODEL = True
            global TEXT_MODEL
            TEXT_MODEL = True
            self.image_adapter = shared_model
            self.text_adapter = shared_model
            params += list(shared_model.parameters())
        else:
            if TEXT_MODEL is False:
                print("*** No text adapter !!!! ***")
                self.text_adapter = None
            elif TEXT_MODEL is True:
                # print("*** Text image_adapter adapter passed in constructor ***")
                if MODEL_USED == "mlp":
                    text_adapter = myMLP().to(device)
                elif MODEL_USED == "dense":
                    text_adapter = myLinearModel().to(device)
                else:
                    print("*** ERROR... ***")
                self.text_adapter = text_adapter
                params += self.text_adapter.parameters()

            if IMAGE_MODEL is False:
                print("*** No IMAGE MODEL !!!! ***")
                self.image_adapter = None
            elif IMAGE_MODEL is True:
                # print("*** Image image_adapter adapter passed in constructor ***")
                if MODEL_USED == "mlp":
                    image_adapter = myMLP().to(device)
                elif MODEL_USED == "dense":
                    image_adapter = myLinearModel().to(device)
                else:
                    print("*** ERROR... ***")
                self.image_adapter = image_adapter
                params += self.image_adapter.parameters()

        print("image adapter", self.image_adapter)  # xxx print summary
        print("text adapter", self.text_adapter)  # xxx print summary
        if len(params)>0:
            print("Creating Adam optimizer...")
            self.optimizer = optim.Adam(params, lr=lr)
            for param_group in self.optimizer.param_groups:
                for name, param in param_group.items():
                    if name == 'params':
                        print('Optimizer parameter names:')
                        for p in param:
                            print(p.shape)
        else:
            self.optimizer = None
        self.val_f1_heat_map = torch.empty((0, 5))
        self.val_auroc_heat_map = torch.empty((0, 5))
        self.test_f1_heat_map = torch.empty((0, 5))
        self.test_auroc_heat_map = torch.empty((0, 5))

    # todo add val metrics
    @torch.no_grad()
    def zero_shot_test_val(self, data_loader, basic_prompts, test="test"):
        y_true = []
        y_pred = []
        with torch.no_grad():
            for embs, labels in tqdm(data_loader, desc="Zero-shot " + test):
                embs = embs.to(self.device)
                labels = labels.to(self.device)
                new_embs = embs
                new_embs = F.normalize(new_embs, dim=-1)

                predicted_labels = torch.zeros(labels.shape[0], 5).to(self.device)

                i = -1
                for label_name in self.class_names:
                    i += 1
                    pos_prompt = self.prompts[label_name]["positive"]
                    neg_prompt = self.prompts[label_name]["negative"]

                    pos_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(pos_prompt, normalize=False)
                    assert pos_prompt_embedding.shape[0] == len(pos_prompt)
                    if not basic_prompts:
                        pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
                    pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(self.device)

                    neg_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(neg_prompt, normalize=False)
                    assert neg_prompt_embedding.shape[0] == len(neg_prompt)
                    if not basic_prompts:
                        neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
                    neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(self.device)

                    pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
                    neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)

                    pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
                    neg_similarities = neg_similarities.reshape(-1, 1)
                    predicted_labels[:, i] = torch.argmax(torch.cat([neg_similarities, pos_similarities], dim=1), dim=1)

                predicted_labels_np = predicted_labels.cpu().numpy()

                y_true.append(labels.cpu().numpy())
                y_pred.append(predicted_labels_np)

        # Concatenate the true and predicted labels
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # Calculate the metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        auroc = roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")
        # precision = precision_score(y_true, y_pred, average="weighted")
        # recall = recall_score(y_true, y_pred, average="weighted")

        # Calculate precision-recall curve for each class
        precision_curve = []
        recall_curve = []
        for i in range(5):
            precision_i, recall_i, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
            precision_curve.append(precision_i)
            recall_curve.append(recall_i)
        if self.writer is not None:
            self.writer.add_scalar(test + "/Comparison Accuracy", accuracy, 0)
            self.writer.add_scalar(test + "/Comparison F1 score", f1, 0)
            self.writer.add_scalar(test + "/Comparison AUROC", auroc, 0)
            for i in range(5):
                self.writer.add_scalar(test + "/Comparison Class Accuracy", accuracy_score(y_true[:, i], y_pred[:, i]),
                                       i)
                self.writer.add_scalar(test + "/Comparison Class Precision",
                                       precision_score(y_true[:, i], y_pred[:, i], average="weighted"), i)
                self.writer.add_scalar(test + "/Comparison Class Recall",
                                       recall_score(y_true[:, i], y_pred[:, i], average="weighted"), i)
            for i in range(5):
                fig = plt.figure()
                plt.plot(recall_curve[i], precision_curve[i], label='Precision-Recall curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve for Class ' + str(i))
                plt.legend(loc="lower left")
                self.writer.add_figure(test + "/Precision-Recall Curve for Class " + str(i), fig)

    @staticmethod
    def preprocessing(chex_competition, xrays_position, single_prompt, batch_size, lr, epochs, loss_name):
        # xxx CHEX COMPETITION
        if chex_competition:
            print("*** CHEX COMPETITION ***")
            class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
            chex_str = "-chex"
        # xxx NOT CHEX competition
        else:
            print("*** NO chex competition ***")
            class_names = ["Pleural Effusion", "Pneumothorax", "Atelectasis", "Pneumonia", "Consolidation"]
            chex_str = ""

        # xxx ALL FRONTAL and NOT
        if xrays_position == "all":
            print("*** ALL FRONTAL and NOT ***")
            train_dataset = torch.load(
                "embeddingDataset\\train\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
            val_dataset = torch.load(
                "embeddingDataset\\val\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
            test_dataset = torch.load(
                "embeddingDataset\\test\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
        # xxx FRONTAL
        elif xrays_position == "frontal":
            print("*** ONLY FRONTAL ***")
            train_dataset = torch.load(
                "embeddingDataset\\train\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")
            val_dataset = torch.load(
                "embeddingDataset\\val\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")
            test_dataset = torch.load(
                "embeddingDataset\\test\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=None, batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4, pin_memory=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, sampler=None, batch_size=1024,
                                                 shuffle=True,
                                                 num_workers=4, pin_memory=True, drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, sampler=None, batch_size=1024,
                                                  shuffle=True,
                                                  num_workers=4, pin_memory=True, drop_last=False)
        print("TrainBS:", batch_size, "Val/Test Batch size default set to 1024")
        if single_prompt:
            str_basic = "-single-prompt"
            prompts = basic_create_prompts(class_names)
        else:
            str_basic = "-mean-prompt"
            prompts = create_prompts(class_names)
        if epochs > 0:
            suffix = "-" + MODEL_USED
            if SHARED:
                suffix += "-SHARED-adapter"
            else:
                if IMAGE_MODEL and TEXT_MODEL:
                    suffix += "-double-adapter"
                elif IMAGE_MODEL:
                    suffix += "-only-image-adapter"
                elif TEXT_MODEL:
                    suffix += "-only-text-adapeter"
            w_path = "./joint-training/joint-train-loss-" + str(loss_name) + "-lr-" + str(
                lr) + "-bs" + str(
                batch_size) + "-ep" + str(
                epochs) + chex_str + str_basic + "-" + str(xrays_position) + suffix
        if epochs == 0:
            if SHARED and IMAGE_MODEL and TEXT_MODEL:
                suffix = "-SHARED-adapter-"+MODEL_USED
            elif not SHARED and not IMAGE_MODEL and not TEXT_MODEL:
                suffix = "-no-head"
            else:
                raise Exception
            print("Attenzione! Zero-shot evaluation!")
            w_path = "./joint-training/zero-shot-model"+ chex_str + str_basic + "-" + str(xrays_position) + suffix
        # w_path = "./joint-training/rapid_check"
        print("writer path:", w_path)
        writer = SummaryWriter(w_path)

        return writer, class_names, train_loader, val_loader, test_loader, prompts

    @staticmethod  # xxx prep for class incremental-one-class
    def preprocessing_class_incremental_one_class(loss_name, chex_competition, xrays_position, basic_prompts,
                                                  batch_size, lr,
                                                  epochs, CONTINUAL_LEARNING, threshold, ratio, tasks_order):
        if CONTINUAL_LEARNING is not None:
            print("**** Gradient clipping ****")
            print("--->", CONTINUAL_LEARNING)

        else:
            print("**** NO Gradient clipping ****")

        # xxx CHEX COMPETITION
        if chex_competition:
            print("*** CHEX COMPETITION ***")
            chex_str = "-chex"
            class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        # xxx NOT CHEX competition
        else:
            print("*** NO chex competition ***")
            class_names = ["Pleural Effusion", "Pneumothorax", "Atelectasis", "Pneumonia", "Consolidation"]
            chex_str = ""

        # xxx ALL FRONTAL and NOT
        if xrays_position == "all":
            print("*** ALL FRONTAL and NOT ***")
            train_dataset = torch.load(
                "embeddingDataset\\train\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
            val_dataset = torch.load(
                "embeddingDataset\\val\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
            test_dataset = torch.load(
                "embeddingDataset\\test\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
        # xxx FRONTAL
        elif xrays_position == "frontal":
            print("*** ONLY FRONTAL ***")
            train_dataset = torch.load(
                "embeddingDataset\\train\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")
            val_dataset = torch.load(
                "embeddingDataset\\val\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")
            test_dataset = torch.load(
                "embeddingDataset\\test\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=None, batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4, pin_memory=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, sampler=None, batch_size=1024,
                                                 shuffle=True,
                                                 num_workers=4, pin_memory=True, drop_last=False)  # 64
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, sampler=None, batch_size=1024,
                                                  shuffle=True,
                                                  num_workers=4, pin_memory=True, drop_last=False)

        # val_loader = Trainer.split_dataloader_by_label(val_loader, batch_size=1024)
        # data_loader = Trainer.split_dataloader_by_label(data_loader, batch_size=1024)
        train_loader = Trainer.concat_to_tensor_dataloader(train_loader)
        train_loader = Trainer.split_dataloader_by_label(train_loader, batch_size=batch_size)
        # xxx to do unmute
        # if True:  # xxx to do check
        #     for i in range(5):
        #         print("Train", i, len(train_loader[i].dataset))
        #         Trainer.count_positive_labels(train_loader[i])
        #         print()
        # for i in range(5):
        #     print("Val", i, len(val_loader[i].dataset))
        #     Trainer.count_positive_labels(val_loader[i])
        #     print()
        # for i in range(5):
        #     print("Test", i, len(data_loader[i].dataset))
        #     Trainer.count_positive_labels(data_loader[i])
        #     print()

        print("TrainBS:", batch_size, "Val/Test Batch size default set to 1024")
        if basic_prompts:
            str_basic = "-NO-prompt"
            prompts = basic_create_prompts(class_names)
        else:
            str_basic = "-mean-prompt"
            prompts = create_prompts(class_names)

        # w_path = "./continual-scenario/fine-tuning-online-lr" + str(lr) + "-bs" + str(batch_size) + "-ep" + str(epochs) + chex_str + str_basic
        w_path = "./class_incremental_one_class_fix/not-adapter-should-forget-" + str(loss_name) + "-lr" + str(
            lr) + "-bs" + str(
            batch_size) + "-ep" + str(
            epochs) + chex_str + str_basic + "-" + str(tasks_order)
        ratio_string = ""
        if ratio:
            ratio_string = "-ratio"
        if CONTINUAL_LEARNING == "profCL":
            w_path = "./class_incremental_one_class_fix/adapter-epoch-level-tr" + str(
                threshold) + ratio_string + "-" + str(loss_name) + "-lr" + str(
                lr) + "-bs" + str(batch_size) + "-ep" + str(
                epochs) + chex_str + str_basic + "-" + str(tasks_order)
        if CONTINUAL_LEARNING == "myCL":
            w_path = "./class_incremental_one_class_fix/adapter-batch-level-tr" + str(
                threshold) + ratio_string + "-" + str(loss_name) + "-lr" + str(
                lr) + "-bs" + str(
                batch_size) + "-ep" + str(
                epochs) + chex_str + str_basic + "-" + str(tasks_order)
        print("summary path", w_path)
        writer = SummaryWriter(w_path)

        return writer, class_names, train_loader, val_loader, test_loader, prompts

    # todo
    @staticmethod  # xxx prep for class incremental-two-class
    def preprocessing_class_incremental_two_class(loss_name, chex_competition, xrays_position, basic_prompts,
                                                  batch_size, lr,
                                                  epochs, CONTINUAL_LEARNING, threshold, ratio, tasks_order):
        if CONTINUAL_LEARNING is not None:
            print("**** Gradient Clipping ****")
            print("--->", CONTINUAL_LEARNING)

        else:
            print("**** NO Gradient Clipping ****")

        # xxx CHEX COMPETITION
        if chex_competition:
            print("*** CHEX COMPETITION ***")
            chex_str = "-chex"
            class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        # xxx NOT CHEX competition
        else:
            print("*** NO chex competition ***")
            class_names = ["Pleural Effusion", "Pneumothorax", "Atelectasis", "Pneumonia", "Consolidation"]
            chex_str = ""

        # xxx ALL FRONTAL and NOT
        if xrays_position == "all":
            print("*** ALL FRONTAL and NOT ***")
            train_dataset = torch.load(
                "embeddingDataset\\train\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
            val_dataset = torch.load(
                "embeddingDataset\\val\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
            test_dataset = torch.load(
                "embeddingDataset\\test\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
        # xxx FRONTAL
        elif xrays_position == "frontal":
            print("*** ONLY FRONTAL ***")
            train_dataset = torch.load(
                "embeddingDataset\\train\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")
            val_dataset = torch.load(
                "embeddingDataset\\val\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")
            test_dataset = torch.load(
                "embeddingDataset\\test\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=None, batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4, pin_memory=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, sampler=None, batch_size=1024,
                                                 shuffle=True,
                                                 num_workers=4, pin_memory=True, drop_last=False)  # 64
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, sampler=None, batch_size=1024,
                                                  shuffle=True,
                                                  num_workers=4, pin_memory=True, drop_last=False)

        # val_loader = Trainer.split_dataloader_by_label(val_loader, batch_size=1024)
        # data_loader = Trainer.split_dataloader_by_label(data_loader, batch_size=1024)
        train_loader = Trainer.concat_to_tensor_dataloader(train_loader)
        train_loader = Trainer.split_dataloader_data_incremental(train_loader, 5)

        # train_loader = Trainer.split_dataloader_by_label(train_loader, batch_size=batch_size)
        # if True:  # xxx to do check
        #     for i in range(5):
        #         print("Train", i, len(train_loader[i].dataset))
        #         Trainer.count_positive_labels(train_loader[i])
        #         print()
        # for i in range(5):
        #     print("Val", i, len(val_loader[i].dataset))
        #     Trainer.count_positive_labels(val_loader[i])
        #     print()
        # for i in range(5):
        #     print("Test", i, len(data_loader[i].dataset))
        #     Trainer.count_positive_labels(data_loader[i])
        #     print()

        print("TrainBS:", batch_size, "Val/Test Batch size default set to 1024")
        if basic_prompts:
            str_basic = "-NO-prompt"
            prompts = basic_create_prompts(class_names)
        else:
            str_basic = "-mean-prompt"
            prompts = create_prompts(class_names)

        # w_path = "./continual-scenario/fine-tuning-online-lr" + str(lr) + "-bs" + str(batch_size) + "-ep" + str(epochs) + chex_str + str_basic
        w_path = "./class_incremental_two_class_F1_tables/mlp-fine-tuning-should-forget-" + str(
            loss_name) + "-lr" + str(
            lr) + "-bs" + str(
            batch_size) + "-ep" + str(
            epochs) + chex_str + str_basic + "-" + str(tasks_order)
        ratio_string = ""
        if ratio:
            ratio_string = "-ratio"
        if CONTINUAL_LEARNING == "profCL":
            w_path = "./class_incremental_two_class_F1_tables/adapter-epoch-level-tr" + str(
                threshold) + ratio_string + "-" + str(loss_name) + "-lr" + str(
                lr) + "-bs" + str(batch_size) + "-ep" + str(
                epochs) + chex_str + str_basic + "-" + str(tasks_order)
        if CONTINUAL_LEARNING == "myCL":
            w_path = "./class_incremental_two_class_F1_tables/adapter-batch-level-tr" + str(
                threshold) + ratio_string + "-" + str(loss_name) + "-lr" + str(
                lr) + "-bs" + str(
                batch_size) + "-ep" + str(
                epochs) + chex_str + str_basic + "-" + str(tasks_order)
        # w_path= "./class_incremental_two_class_F1_tables/debug"
        print("summary path", w_path)
        writer = SummaryWriter(w_path)

        return writer, class_names, train_loader, val_loader, test_loader, prompts

    @staticmethod  # xxx prep for data-incremental
    def preprocessing_data_incremental(chex_competition, xrays_position, basic_prompts, batch_size, lr,
                                       epochs, CONTINUAL_LEARNING, threshold, ratio):

        if CONTINUAL_LEARNING is not None:
            print("**** Gradient Clipping ****")
            print("--->", CONTINUAL_LEARNING)

        else:
            print("**** NO Gradient Clipping ****")

        # xxx CHEX COMPETITION
        if chex_competition:
            print("*** CHEX COMPETITION ***")
            chex_str = "-chex"
            class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        # xxx NOT CHEX competition
        else:
            print("*** NO chex competition ***")
            class_names = ["Pleural Effusion", "Pneumothorax", "Atelectasis", "Pneumonia", "Consolidation"]
            chex_str = ""

        # xxx ALL FRONTAL and NOT
        if xrays_position == "all":
            print("*** ALL FRONTAL and NOT ***")
            train_dataset = torch.load(
                "embeddingDataset\\train\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
            val_dataset = torch.load(
                "embeddingDataset\\val\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
            test_dataset = torch.load(
                "embeddingDataset\\test\\512" + chex_str + "-not-normalize\\embeddings_dataset_final_old.pt")
        # xxx FRONTAL
        elif xrays_position == "frontal":
            print("*** ONLY FRONTAL ***")
            train_dataset = torch.load(
                "embeddingDataset\\train\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")
            val_dataset = torch.load(
                "embeddingDataset\\val\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")
            test_dataset = torch.load(
                "embeddingDataset\\test\\512" + chex_str + "-not-normalize-frontal\\embeddings_dataset_final_old.pt")

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=None, batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4, pin_memory=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, sampler=None, batch_size=1024,
                                                 shuffle=True,
                                                 num_workers=4, pin_memory=True, drop_last=False)  # 64
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, sampler=None, batch_size=1024,
                                                  shuffle=True,
                                                  num_workers=4, pin_memory=True, drop_last=False)

        train_loader = Trainer.split_dataloader_data_incremental(train_loader, epochs)

        # Trainer.print_dataloader_stats(train_loader) xxx to do unmute

        print("TrainBS:", batch_size, "Val/Test Batch size default set to 1024")
        if basic_prompts:
            str_basic = "-NO-prompt"
            prompts = basic_create_prompts(class_names)
        else:
            str_basic = "-mean-prompt"
            prompts = create_prompts(class_names)

        # w_path = "./continual-scenario/fine-tuning-online-lr" + str(lr) + "-bs" + str(batch_size) + "-ep" + str(epochs) + chex_str + str_basic
        w_path = "./data_incremental/not-adapter-lr" + str(lr) + "-bs" + str(batch_size) + "-ep" + str(
            epochs) + chex_str + str_basic
        ratio_string = ""
        if ratio:
            ratio_string = "-ratio"
        if CONTINUAL_LEARNING == "profCL":
            w_path = "./data_incremental/adapter-epoch-level-tr" + str(threshold) + ratio_string + "-lr" + str(
                lr) + "-bs" + str(batch_size) + "-ep" + str(
                epochs) + chex_str + str_basic
        if CONTINUAL_LEARNING == "myCL":
            w_path = "./data_incremental/adapter-batch-level-tr" + str(threshold) + ratio_string + "-lr" + str(
                lr) + "-bs" + str(
                batch_size) + "-ep" + str(
                epochs) + chex_str + str_basic
        print("writer path:", w_path)
        writer = SummaryWriter(w_path)

        return writer, class_names, train_loader, val_loader, test_loader, prompts

    # xxx works for normal and for data-incremental
    def train(self, train_loader, criterion, epoch, CONTINUAL_LEARNING=None, threshold=None,
              scheduler=None):
        batch_idx = 0
        if IMAGE_MODEL:
            self.image_adapter.train()
        if TEXT_MODEL:
            self.text_adapter.train()
        for embs, labels in tqdm(train_loader, desc="Fine-tuning on chexpert, Epoch " + str(epoch)):
            if CONTINUAL_LEARNING == "myCL":
                # make a copy of the original image_adapter before starting the training loop
                model_copy = copy.deepcopy(self.image_adapter)
                # define the threshold for parameter updates
                n_reset = 0
                n_updated = 0
            if self.loss_name == "opzione2" or self.loss_name == "opzione2variant" or self.loss_name == "bce":
                loss = 0.0
            self.optimizer.zero_grad()
            batch_idx += 1
            embs = embs.to(self.device)
            labels = labels.to(self.device)
            if IMAGE_MODEL:
                new_embs = self.image_adapter(embs)
            else:
                new_embs = embs
            new_embs = F.normalize(new_embs, dim=-1)

            if not self.loss_name == "bce":
                logits = torch.zeros(labels.shape[0], 5).to(self.device)
            else:  # if self.loss_name == "bce":
                logits = torch.empty(0, 2).to(self.device)
                targets = torch.empty(0, 2).to(self.device)

            i = -1
            for label_name in self.class_names:
                i += 1
                pos_prompt = self.prompts[label_name]["positive"]
                neg_prompt = self.prompts[label_name]["negative"]

                pos_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(pos_prompt, normalize=False)
                if TEXT_MODEL:
                    pos_prompt_embedding = pos_prompt_embedding.to(self.device)
                    pos_prompt_embedding = self.text_adapter(pos_prompt_embedding)
                assert pos_prompt_embedding.shape[0] == len(pos_prompt)
                if not self.basic_prompts:
                    pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
                pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(self.device)

                neg_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(neg_prompt, normalize=False)
                if TEXT_MODEL:
                    neg_prompt_embedding = neg_prompt_embedding.to(self.device)
                    neg_prompt_embedding = self.text_adapter(neg_prompt_embedding)
                assert neg_prompt_embedding.shape[0] == len(neg_prompt)
                if not self.basic_prompts:
                    neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
                neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(self.device)

                # Calculate the similarities between the image and the positive and negative prompts
                pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
                neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)

                if self.loss_name == "standard":
                    logits[:, i] = pos_similarities - neg_similarities
                elif self.loss_name == "bce":
                    logits = torch.cat(
                        [logits, torch.cat([neg_similarities.unsqueeze(-1), pos_similarities.unsqueeze(-1)], dim=1)],
                        dim=0)
                    mask = labels[:, i] == 1
                    tmp_targets = torch.zeros(labels.shape[0], 2).to(self.device)
                    tmp_targets[mask, 1] = 1
                    tmp_targets[~mask, 0] = 1
                    targets = torch.cat([targets, tmp_targets], dim=0)

                elif self.loss_name == "opzione2":  # todo optimze
                    for j in range(labels.shape[0]):
                        if labels[j][i] == 0:
                            loss += -neg_similarities[j]
                        elif labels[j][i] == 1:
                            loss += -pos_similarities[j]
                    loss = 1 + loss / labels.shape[0]
                elif self.loss_name == "opzione2variant":  # todo optimze
                    for j in range(labels.shape[0]):
                        if labels[j][i] == 0:
                            loss += -neg_similarities[j] + pos_similarities[j]
                        elif labels[j][i] == 1:
                            loss += -pos_similarities[j] + neg_similarities[j]
                    loss = 1 + loss / labels.shape[0]

            if self.change_labels:
                labels = change_values(labels)
            if self.loss_name == "standard":
                loss = criterion(logits, labels)
            elif self.loss_name == "bce":
                loss = criterion(logits, targets)
            elif self.loss_name == "opzione2":
                pass
            elif self.loss_name == "opzione2variant":
                pass
            loss.backward()
            self.optimizer.step()
            iteration = (epoch - 1) * len(train_loader) + batch_idx

            if CONTINUAL_LEARNING == "myCL":
                for (name1, param1), (name2, param2) in zip(self.image_adapter.named_parameters(),
                                                            model_copy.named_parameters()):
                    # compare the values of the individual weights
                    diff = torch.abs(param1 - param2)
                    minimum = diff.min()
                    maximum = diff.max()
                    # compute the threshold value
                    to_reset = minimum + threshold * (maximum - minimum)
                    mask = diff < to_reset

                    n_reset += torch.sum(torch.eq(mask, True))
                    n_updated += torch.sum(torch.eq(mask, False))
                    # reset the updated weights to the old values
                    param1.data[mask] = param2.data[mask]
                print()

                print("number of resets:", n_reset.item(), "number of updates:", n_updated.item(), "percentage resets",
                      n_reset.item() / (n_reset.item() + n_updated.item()))
                self.writer.add_scalar("train/monitor-resets/resets", n_reset.item(), iteration)
                self.writer.add_scalar("train/monitor-resets/updates", n_updated.item(), iteration)
                self.writer.add_scalar("train/monitor-resets/percentage resets",
                                       n_reset.item() / (n_reset.item() + n_updated.item()),
                                       iteration)

            if self.writer is not None:
                self.writer.add_scalar('train/Loss', loss.item(), iteration)
            if scheduler is not None:
                scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/LR', current_lr, iteration)

    # xxx works for class-incremental one class and both class
    def train_class_incremental(self, train_loader, optimizer, criterion, epoch, basic_prompts, current_task,
                                CONTINUAL_LEARNING=None, threshold=None, last_batch=0):
        # xxx ONE EPOCH TRAIN
        batch_idx = last_batch
        self.image_adapter.train()
        for embs, labels in tqdm(train_loader,
                                 desc="Fine-tuning on task " + str(current_task) + ", Epoch " + str(epoch)):
            if CONTINUAL_LEARNING == "myCL":
                # make a copy of the original image_adapter before starting the training loop
                model_copy = copy.deepcopy(self.image_adapter)
                # define the threshold for parameter updates
                n_reset = 0
                n_updated = 0
            if self.loss_name == "opzione2" or self.loss_name == "opzione2variant":
                loss = 0.0
            optimizer.zero_grad()
            batch_idx += 1
            embs = embs.to(self.device)
            labels = labels.to(self.device)
            labels = labels[:, current_task]
            new_embs = self.image_adapter(embs)
            new_embs = F.normalize(new_embs, dim=-1)

            logits = torch.zeros(labels.shape[0]).to(self.device)

            # for label_name in self.class_names:
            label_name = self.class_names[current_task]
            pos_prompt = self.prompts[label_name]["positive"]
            neg_prompt = self.prompts[label_name]["negative"]

            pos_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(pos_prompt, normalize=False)
            assert pos_prompt_embedding.shape[0] == len(pos_prompt)
            if not basic_prompts:
                pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
            pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(self.device)

            neg_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(neg_prompt, normalize=False)
            assert neg_prompt_embedding.shape[0] == len(neg_prompt)
            if not basic_prompts:
                neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
            neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(self.device)

            # Calculate the similarities between the image and the positive and negative prompts
            pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
            neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)

            # xxx pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
            # xxx neg_similarities = neg_similarities.reshape(-1, 1)
            # xxx NON E' DERIVABILE LOL Take the maximum similarity as the predicted label
            # xxx predicted_labels[:, i] = torch.argmax(torch.cat([neg_similarities, pos_similarities], dim=1), dim=1)
            if self.loss_name == "standard":
                logits = pos_similarities - neg_similarities  # XXX grandissima differnza

            elif self.loss_name == "opzione2":  # todo optimze
                for j in range(labels.shape[0]):
                    if labels[j] == 0:
                        loss += -neg_similarities[j]
                    elif labels[j] == 1:
                        loss += -pos_similarities[j]
                loss = 1 + loss / labels.shape[0]
            elif self.loss_name == "opzione2variant":
                for j in range(labels.shape[0]):
                    if labels[j] == 0:
                        loss += -neg_similarities[j] + pos_similarities[j]
                    elif labels[j] == 1:
                        loss += -pos_similarities[j] + neg_similarities[j]
                loss = 1 + loss / labels.shape[0]
            # Compute loss and backpropagate
            # todo fare tutte le loss
            # loss figa con labels -2, 2 che ipoteticamnete spara pos a 1 neg a -1 e viceversa
            if self.change_labels:
                labels = change_values(labels)
            # loss = criterion(predicted_labels, labels)

            if self.loss_name == "standard":
                loss = criterion(logits, labels)
            elif self.loss_name == "opzione2":
                pass
            elif self.loss_name == "opzione2variant":
                pass
            loss.backward()
            optimizer.step()

            iteration = batch_idx
            if CONTINUAL_LEARNING == "myCL":
                for (name1, param1), (name2, param2) in zip(self.image_adapter.named_parameters(),
                                                            model_copy.named_parameters()):
                    # compare the values of the individual weights
                    diff = torch.abs(param1 - param2)
                    minimum = diff.min()
                    maximum = diff.max()
                    # compute the threshold value
                    to_reset = minimum + threshold * (maximum - minimum)
                    mask = diff < to_reset

                    n_reset += torch.sum(torch.eq(mask, True))
                    n_updated += torch.sum(torch.eq(mask, False))
                    # reset the updated weights to the old values
                    param1.data[mask] = param2.data[mask]
                print()
                # iteration = (epoch - 1) * len(train_loader) + batch_idx

                print("number of resets:", n_reset.item(), "number of updates:", n_updated.item(), "percentage resets",
                      n_reset.item() / (n_reset.item() + n_updated.item()))
                self.writer.add_scalar("monitor-resets/resets", n_reset.item(), iteration)
                self.writer.add_scalar("monitor-resets/updates", n_updated.item(), iteration)
                self.writer.add_scalar("monitor-resets/percentage resets",
                                       n_reset.item() / (n_reset.item() + n_updated.item()),
                                       iteration)
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss', loss.item(), iteration)
            # if scheduler is not None:
            #     scheduler.step()
            #     current_lr = optimizer.param_groups[0]['lr']
            #     self.writer.add_scalar('LR', current_lr, iteration)
        return iteration

    @torch.no_grad()
    def val(self, val_loader, criterion, epoch, epochs, mode=None, tasks_order=None):
        # xxx ONE EPOCH VAL
        batch_idx = 0
        y_true = []
        y_pred = []
        if IMAGE_MODEL:
            self.image_adapter.eval()
        if TEXT_MODEL:
            self.text_adapter.eval()
        with torch.no_grad():
            for embs, labels in tqdm(val_loader, desc="Validating on chexpert, Epoch " + str(epoch)):
                if self.loss_name == "opzione2" or self.loss_name == "opzione2variant":
                    loss = 0.0
                batch_idx += 1
                embs = embs.to(self.device)
                labels = labels.to(self.device)
                if IMAGE_MODEL:
                    new_embs = self.image_adapter(embs)
                else:
                    new_embs = embs
                new_embs = F.normalize(new_embs, dim=-1)

                predicted_labels = torch.zeros(labels.shape[0], 5).to(self.device)

                if not self.loss_name == "bce":
                    logits = torch.zeros(labels.shape[0], 5).to(self.device)
                else:  # if self.loss_name == "bce":
                    logits = torch.empty(0, 2).to(self.device)
                    targets = torch.empty(0, 2).to(self.device)

                i = -1
                for label_name in self.class_names:
                    i += 1
                    pos_prompt = self.prompts[label_name]["positive"]
                    neg_prompt = self.prompts[label_name]["negative"]

                    pos_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(pos_prompt, normalize=False)
                    if TEXT_MODEL:
                        pos_prompt_embedding = pos_prompt_embedding.to(self.device)
                        pos_prompt_embedding = self.text_adapter(pos_prompt_embedding)
                    assert pos_prompt_embedding.shape[0] == len(pos_prompt)
                    if not self.basic_prompts:
                        pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
                    pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(self.device)

                    neg_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(neg_prompt, normalize=False)
                    if TEXT_MODEL:
                        neg_prompt_embedding = neg_prompt_embedding.to(self.device)
                        neg_prompt_embedding = self.text_adapter(neg_prompt_embedding)
                    assert neg_prompt_embedding.shape[0] == len(neg_prompt)
                    if not self.basic_prompts:
                        neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
                    neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(self.device)

                    pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
                    neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)

                    if self.loss_name == "standard":
                        logits[:, i] = pos_similarities - neg_similarities  # XXX grandissima differnza
                    elif self.loss_name == "bce":
                        logits = torch.cat([logits,
                                            torch.cat([neg_similarities.unsqueeze(-1), pos_similarities.unsqueeze(-1)],
                                                      dim=1)], dim=0)
                        # tmp_labels = labels[:, i]
                        # tmp_labels = Trainer.convert_1d_to_2d(tmp_labels)
                        # tmp_labels = torch.tensor(tmp_labels).to(self.device)
                        mask = labels[:, i] == 1
                        tmp_targets = torch.zeros(labels.shape[0], 2).to(self.device)
                        tmp_targets[mask, 1] = 1
                        tmp_targets[~mask, 0] = 1
                        targets = torch.cat([targets, tmp_targets], dim=0)
                    elif self.loss_name == "opzione2":  # todo optimize
                        for j in range(labels.shape[0]):
                            if labels[j][i] == 0:
                                loss += -neg_similarities[j]
                            elif labels[j][i] == 1:
                                loss += -pos_similarities[j]
                        loss = 1 + loss / labels.shape[0]
                    elif self.loss_name == "opzione2variant":  # todo optimize
                        for j in range(labels.shape[0]):
                            if labels[j][i] == 0:
                                loss += -neg_similarities[j] + pos_similarities[j]
                            elif labels[j][i] == 1:
                                loss += -pos_similarities[j] + neg_similarities[j]
                        loss = 1 + loss / labels.shape[0]
                    pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
                    neg_similarities = neg_similarities.reshape(-1, 1)
                    # xxx NON E' DERIVABILE LOL Take the maximum similarity as the predicted label
                    predicted_labels[:, i] = torch.argmax(torch.cat([neg_similarities, pos_similarities], dim=1),
                                                          dim=1)

                # Compute loss and backpropagate
                if self.change_labels:
                    tmp = labels
                    labels = change_values(labels)
                # loss = criterion(loss_predicted_labels, loss_labels)  # todo occhio alla differenza loss_ non loss_
                # loss = criterion(logits, labels)  # todo occhio alla differenza loss_ non loss_
                if self.loss_name == "standard":
                    loss = criterion(logits, labels)
                elif self.loss_name == "bce":
                    loss = criterion(logits, targets)
                elif self.loss_name == "opzione2":
                    pass
                elif self.loss_name == "opzione2variant":
                    pass
                iteration = (epoch - 1) * len(val_loader) + batch_idx
                if self.writer is not None:
                    self.writer.add_scalar('val/Loss', loss.item(), iteration)

                # Convert the predicted labels to a numpy array
                predicted_labels_np = predicted_labels.cpu().numpy()

                # Append the true and predicted labels to the lists
                if not self.change_labels:
                    y_true.append(labels.cpu().numpy())
                if self.change_labels:
                    y_true.append(tmp.cpu().numpy())
                y_pred.append(predicted_labels_np)

        # Concatenate the true and predicted labels
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # Calculate the metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        auroc_macro = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
        auroc_weighted = roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")

        # Calculate precision-recall curve for each class
        precision_curve = []
        recall_curve = []
        for i in range(5):
            precision_i, recall_i, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
            precision_curve.append(precision_i)
            recall_curve.append(recall_i)

        if self.writer is not None:
            self.writer.add_scalar("val/Accuracy", accuracy, epoch)
            self.writer.add_scalar("val/F1-macro score", f1_macro, epoch)
            self.writer.add_scalar("val/F1-weighted score", f1_weighted, epoch)
            self.writer.add_scalar("val/AUROC-macro", auroc_macro, epoch)
            self.writer.add_scalar("val/AUROC-weighted", auroc_weighted, epoch)

            tmp_f1 = torch.zeros(1, 5)
            tmp_auroc = torch.zeros(1, 5)
            for i in range(5):
                # tmp_f1[0, i] = f1_score(y_true[:, i], y_pred[:, i], average="weighted")
                tmp_f1[0, i] = f1_score(y_true[:, i], y_pred[:, i])
                # tmp_auroc[0, i] = roc_auc_score(y_true[:, i], y_pred[:, i], average="weighted",
                #                                                    multi_class="ovr")
                tmp_auroc[0, i] = roc_auc_score(y_true[:, i], y_pred[:, i])

            self.val_f1_heat_map = torch.cat([self.val_f1_heat_map, tmp_f1], dim=0)
            self.val_auroc_heat_map = torch.cat([self.val_auroc_heat_map, tmp_auroc], dim=0)

            if epoch == epochs:
                self.val_f1_heat_map = numpy.array(self.val_f1_heat_map)
                fig, ax = plt.subplots()
                if mode == "class pos-neg":
                    im, cbar = heatmap(self.val_f1_heat_map, [self.class_names[i] for i in tasks_order],
                                       [self.class_names[i] for i in tasks_order], ax=ax,
                                       cmap="YlGn", cbarlabel="F1 score")
                elif mode == "joint":
                    im, cbar = heatmap(self.val_f1_heat_map, [i for i in range(1, epochs + 1)],
                                       [self.class_names[i] for i in range(0, 5)], ax=ax,
                                       cmap="YlGn", cbarlabel="F1 score")
                texts = annotate_heatmap(im, valfmt="{x:.2f}")
                fig.tight_layout()
                # plt.show()
                if mode == "class pos-neg":
                    self.writer.add_figure('val/class pos-neg incremental/F1 score Heatmap', fig)
                elif mode == "joint":
                    self.writer.add_figure('val/joint train/F1 score Heatmap', fig)

                self.val_auroc_heat_map = numpy.array(self.val_auroc_heat_map)
                fig, ax = plt.subplots()
                if mode == "class pos-neg":
                    im, cbar = heatmap(self.val_auroc_heat_map, [self.class_names[i] for i in tasks_order],
                                       [self.class_names[i] for i in tasks_order], ax=ax,
                                       cmap="YlGn", cbarlabel="AUROC score")
                elif mode == "joint":
                    im, cbar = heatmap(self.val_auroc_heat_map, [i for i in range(1, epochs + 1)],
                                       [self.class_names[i] for i in range(0, 5)], ax=ax,
                                       cmap="YlGn", cbarlabel="AUROC score")
                texts = annotate_heatmap(im, valfmt="{x:.2f}")
                fig.tight_layout()
                # plt.show()
                if mode == "class pos-neg":
                    self.writer.add_figure('val/class pos-neg incremental/AUROC score Heatmap', fig)
                elif mode == "joint":
                    self.writer.add_figure('val/joint train/AUROC score Heatmap', fig)

                for i in range(5):
                    self.writer.add_scalar("val/Class Accuracy ep" + str(epoch),
                                           accuracy_score(y_true[:, i], y_pred[:, i]),
                                           i)
                    self.writer.add_scalar("val/Class Precision ep" + str(epoch),
                                           precision_score(y_true[:, i], y_pred[:, i]), i)
                    self.writer.add_scalar("val/Class Recall ep" + str(epoch),
                                           recall_score(y_true[:, i], y_pred[:, i]), i)
        # for i in range(5):
        #     fig = plt.figure()
        #     plt.plot(recall_curve[i], precision_curve[i], label='Precision-Recall curve')
        #     plt.xlabel('Recall')
        #     plt.ylabel('Precision')
        #     plt.title('Precision-Recall Curve for Class ' + str(i))
        #     plt.legend(loc="lower left")
        #     writer.add_figure('Precision-Recall Curve for Class ' + str(i), fig)

    @torch.no_grad()
    def test(self, test_loader, criterion, epoch, epochs, mode=None, tasks_order=None):
        # XXX TEST
        y_true = []
        y_pred = []
        if IMAGE_MODEL:
            self.image_adapter.eval()
        if TEXT_MODEL:
            self.text_adapter.eval()
        with torch.no_grad():
            for embs, labels in tqdm(test_loader, desc="Testing on chexpert"):

                # image0_to_plt = embs[0]
                # image0_to_plt = image0_to_plt.permute(1, 2, 0)
                # # Plot the RGB tensor
                # plt.imshow(image0_to_plt)
                # plt.show()

                embs = embs.to(self.device)
                # embs = torch.rand_like(embs).to(self.device)
                labels = labels.to(self.device)
                if IMAGE_MODEL:
                    new_embs = self.image_adapter(embs)
                else:
                    new_embs = embs
                new_embs = F.normalize(new_embs, dim=-1)

                predicted_labels = torch.zeros(labels.shape[0], 5).to(self.device)

                # Loop through each label
                i = -1
                for label_name in self.class_names:
                    i += 1
                    # Get the positive and negative prompts for the label
                    pos_prompt = self.prompts[label_name]["positive"]
                    neg_prompt = self.prompts[label_name]["negative"]

                    # pos_prompt = pos_prompt.to(device)
                    # neg_prompt = neg_prompt.to(device)
                    # Encode the positive and negative prompts
                    pos_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(pos_prompt, normalize=False)
                    if TEXT_MODEL:
                        pos_prompt_embedding = pos_prompt_embedding.to(self.device)
                        pos_prompt_embedding = self.text_adapter(pos_prompt_embedding)
                    assert pos_prompt_embedding.shape[0] == len(pos_prompt)
                    if not self.basic_prompts:
                        pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
                    pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(self.device)

                    neg_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(neg_prompt, normalize=False)
                    if TEXT_MODEL:
                        neg_prompt_embedding = neg_prompt_embedding.to(self.device)
                        neg_prompt_embedding = self.text_adapter(neg_prompt_embedding)
                    assert neg_prompt_embedding.shape[0] == len(neg_prompt)
                    if not self.basic_prompts:
                        neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
                    neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(self.device)

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
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")
        auroc_macro = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovr")
        auroc_weighted = roc_auc_score(y_true, y_pred, average="weighted", multi_class="ovr")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")

        # Calculate precision-recall curve for each class
        precision_curve = []
        recall_curve = []
        for i in range(5):
            precision_i, recall_i, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
            precision_curve.append(precision_i)
            recall_curve.append(recall_i)

        if self.writer is not None:
            self.writer.add_scalar("test/Accuracy", accuracy, epoch)
            self.writer.add_scalar("test/F1-macro score", f1_macro, epoch)
            self.writer.add_scalar("test/F1-weighted score", f1_weighted, epoch)
            self.writer.add_scalar("test/AUROC-macro", auroc_macro, epoch)
            self.writer.add_scalar("test/AUROC-weighted", auroc_weighted, epoch)

            tmp_f1 = torch.zeros(1, 5)
            tmp_auroc = torch.zeros(1, 5)
            for i in range(5):
                # tmp_f1[0, i] = f1_score(y_true[:, i], y_pred[:, i], average="weighted")
                tmp_f1[0, i] = f1_score(y_true[:, i], y_pred[:, i])
                # tmp_auroc[0, i] = roc_auc_score(y_true[:, i], y_pred[:, i], average="weighted",
                #                                                    multi_class="ovr")
                tmp_auroc[0, i] = roc_auc_score(y_true[:, i], y_pred[:, i])

            self.test_f1_heat_map = torch.cat([self.test_f1_heat_map, tmp_f1], dim=0)
            self.test_auroc_heat_map = torch.cat([self.test_auroc_heat_map, tmp_auroc], dim=0)
            # if mode == "class":
            #     for i in range(5):
            #         self.writer.add_scalar(
            #             "test/Class Accuracy/after task-" + str(epoch) + "-tt-" + str(tasks_order),
            #             accuracy_score(y_true[:, i], y_pred[:, i]), i)
            #         self.writer.add_scalar(
            #             "test/Class Precision/after task-" + str(epoch) + "-tt-" + str(tasks_order),
            #             precision_score(y_true[:, i], y_pred[:, i], average="weighted"), i)
            #         self.writer.add_scalar(
            #             "test/Class Recall/after task-" + str(epoch) + "-tt-" + str(tasks_order),
            #             recall_score(y_true[:, i], y_pred[:, i], average="weighted"), i)
            if epoch == epochs:
                self.test_f1_heat_map = numpy.array(self.test_f1_heat_map)
                fig, ax = plt.subplots()
                if mode == "class pos-neg":
                    im, cbar = heatmap(self.test_f1_heat_map, [self.class_names[i] for i in tasks_order],
                                       [self.class_names[i] for i in tasks_order], ax=ax,
                                       cmap="YlGn", cbarlabel="F1 score")
                elif mode == "joint":
                    im, cbar = heatmap(self.test_f1_heat_map, [i for i in range(1, epochs+1)],
                                       [self.class_names[i] for i in range(0, 5)], ax=ax,
                                       cmap="YlGn", cbarlabel="F1 score")
                texts = annotate_heatmap(im, valfmt="{x:.2f}")
                fig.tight_layout()
                # plt.show()
                if mode == "class pos-neg":
                    self.writer.add_figure('test/class pos-neg incremental/F1 score Heatmap', fig)
                elif mode == "joint":
                    self.writer.add_figure('test/joint train/F1 score Heatmap', fig)

                self.test_auroc_heat_map = numpy.array(self.test_auroc_heat_map)
                fig, ax = plt.subplots()
                if mode == "class pos-neg":
                    im, cbar = heatmap(self.test_auroc_heat_map, [self.class_names[i] for i in tasks_order],
                                       [self.class_names[i] for i in tasks_order], ax=ax,
                                       cmap="YlGn", cbarlabel="AUROC score")
                elif mode == "joint":
                    im, cbar = heatmap(self.test_auroc_heat_map, [i for i in range(1, epochs+1)],
                                       [self.class_names[i] for i in range(0, 5)], ax=ax,
                                       cmap="YlGn", cbarlabel="AUROC score")
                texts = annotate_heatmap(im, valfmt="{x:.2f}")
                fig.tight_layout()
                # plt.show()
                if mode == "class pos-neg":
                    self.writer.add_figure('test/class pos-neg incremental/AUROC score Heatmap', fig)
                elif mode == "joint":
                    self.writer.add_figure('test/joint train/AUROC score Heatmap', fig)

                for i in range(5):
                    self.writer.add_scalar("test/Class Accuracy", accuracy_score(y_true[:, i], y_pred[:, i]), i)
                    self.writer.add_scalar("test/Class Precision",
                                           precision_score(y_true[:, i], y_pred[:, i]), i)
                    self.writer.add_scalar("test/Class Recall",
                                           recall_score(y_true[:, i], y_pred[:, i]), i)
            for i in range(5):
                fig = plt.figure()
                plt.plot(recall_curve[i], precision_curve[i], label='Precision-Recall curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve for Class ' + str(i))
                plt.legend(loc="lower left")
                self.writer.add_figure('test/Precision-Recall Curve for Class ' + str(i), fig)

    @staticmethod  # xxx for class-incremental
    def split_dataloader_by_label(dataloader, batch_size):
        # Initialize an empty list to store the dataloaders
        dataloaders = []
        # Loop over the number of labels
        for i in range(5):
            # Initialize an empty list to store the indices of the samples with the i-th label
            indices = []
            # Loop over the datasets in the ConcatDataset
            if isinstance(dataloader.dataset, torch.utils.data.TensorDataset):
                # For a TensorDataset, directly access the tensors attribute
                mask = dataloader.dataset.tensors[1][:, i] == 1
                indices = torch.where(mask)[0]
            else:
                raise ValueError("Unsupported dataset type")
            # Create a subset of the dataset with the samples that have the i-th label
            subset = torch.utils.data.Subset(dataloader.dataset, indices)
            # Create a new sampler to shuffle the subset
            sampler = RandomSampler(subset)
            # Create a new dataloader for the subset with the sampler
            subset_dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, sampler=sampler,
                                                            num_workers=4, pin_memory=True, drop_last=False)
            # Add the new dataloader to the list
            dataloaders.append(subset_dataloader)
        # Return the list of dataloaders
        return dataloaders

    @staticmethod  # xxx for data-incremental
    def split_dataloader_data_incremental(dataloader, n):
        """
        Splits a PyTorch DataLoader object with a ConcatDataset into N smaller DataLoader objects of equal size.

        Args:
            dataloader (DataLoader): The PyTorch DataLoader object to split.
            n (int): The number of smaller DataLoader objects to split into.
            sampler (Sampler, optional): The sampler to use for each of the smaller DataLoader objects. Defaults to None,
                which will result in using the same sampler as the original DataLoader object.

        Returns:
            A list of N PyTorch DataLoader objects of equal size.
        """
        dataset = dataloader.dataset
        num_samples = len(dataset)
        subset_size = math.ceil(num_samples / n)
        subsets = [Subset(dataset, range(i * subset_size, min((i + 1) * subset_size, num_samples))) for i in range(n)]

        dataloaders = [
            DataLoader(subset, batch_size=dataloader.batch_size, sampler=RandomSampler(subset), num_workers=4,
                       pin_memory=True, drop_last=False) for subset in subsets]
        return dataloaders

    @staticmethod  # for class-incremental info
    def count_positive_labels(dataloader):
        # Initialize a list to keep track of the number of positive labels for each label
        num_positive_labels = [0] * 5

        # Iterate over the data in the dataloader
        for data in dataloader:
            # Get the binary multi-labels from the data
            labels = data[1]

            # Iterate over each label and increment the count of positive labels if the label is positive
            for i in range(5):
                num_positive_labels[i] += labels[:, i].sum()

        # Print the number of positive labels for each label
        for i in range(5):
            print(f"Label {i}: {num_positive_labels[i]}")

    @staticmethod  # fro class-incremental (train dataloader has a concatdataset)
    def concat_to_tensor_dataloader(dataloader):
        # Concatenate all the datasets in the original dataloader
        concat_dataset = ConcatDataset(dataloader.dataset.datasets)

        # Create a list of tensors for the input and target data
        inputs = []
        targets = []
        for dataset in concat_dataset.datasets:
            inputs.append(dataset.tensors[0])
            targets.append(dataset.tensors[1])

        # Create a TensorDataset from the concatenated data
        tensor_dataset = TensorDataset(torch.cat(inputs), torch.cat(targets))

        # Create a new DataLoader using the TensorDataset
        tensor_dataloader = DataLoader(tensor_dataset, batch_size=dataloader.batch_size,
                                       num_workers=dataloader.num_workers, pin_memory=dataloader.pin_memory,
                                       drop_last=dataloader.drop_last)

        return tensor_dataloader

    @staticmethod  # data-incremental info sulla distribuzione labels
    def print_dataloader_stats(dataloaders):
        """
        Prints statistics for each DataLoader in a list of PyTorch DataLoader objects.

        Args:
            dataloaders (list): A list of PyTorch DataLoader objects to print statistics for.
        """
        for i, dataloader in enumerate(dataloaders):
            print(f"Dataloader {i}:")
            print(f"Length: {len(dataloader.dataset)}")
            label_counts = [0] * 5
            for x, labels in dataloader:
                for j in range(5):
                    label_counts[j] += (labels[:, j] == 1).sum().item()
            print(f"Label counts: {label_counts}")


def change_values(tensor):
    """
    Takes a 2D torch tensor of float32 with 0 and 1 values and changes 1 to 2 and 0 to -2.
    Returns the modified tensor as a tensor of float32.
    """
    # Check if the input tensor is a 2D tensor
    if len(tensor.shape) != 2:
        raise ValueError("Input tensor must be a 2D tensor.")

    # Create a copy of the input tensor
    new_tensor = tensor.clone()

    # Replace 1 with 2 and 0 with -2
    new_tensor[tensor == 1] = 1  # 2
    new_tensor[tensor == 0] = -1  # -2

    # Convert the tensor to float32
    new_tensor = new_tensor.float()

    return new_tensor
