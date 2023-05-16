import copy
import math
import warnings

import numpy
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_curve, roc_curve, average_precision_score
from torch.utils.data import Dataset, DataLoader, RandomSampler, ConcatDataset, TensorDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import pairwise_cosine_similarity
from torchvision.io import read_image
from tqdm import tqdm

from DataRetrieval import basic_create_prompts, create_prompts
from HeatMapPlotter import heatmap, annotate_heatmap
from health_multimodal.text import get_cxr_bert_inference
from models import myLinearModel, myMLP

# zero shot o SHARED true, IMAGE true TEXT true
# oppure con SHARED false, IMAGE false e TEXT false
SHARED = False  # True,  False # xxx shared true mette gli altri due a true <3
# con shared false invece si può fare che ci pare
IMAGE_MODEL = True  # True, False
TEXT_MODEL = True  # True, False
MODEL_USED = "mlp"  # mlp, dense, "no-head"

OPTIM = "adam"  # sgd

MAX_EMB = False
NEW_PROMPTS = False

TRAIN_LOGIT_DIFF = True  # False <--> only POS
PRED_LOGIT_DIFF = True  # False <--> only POS

# UNDER_SAMPLE = False
CHANGE_LABELS = False


def filter_dataloader_multiclass(dataloader):
    filtered_dataset = []
    N = 200
    counts = [0, 0, 0, 0, 0]
    for input, label in dataloader.dataset:
        if label.tolist() == [1.0, 0.0, 0.0, 0.0, 0.0] and counts[0] < N:
            counts[0] += 1
            filtered_dataset.append((input, label))
        if label.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0] and counts[1] < N:
            counts[1] += 1
            filtered_dataset.append((input, label))
        if label.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0] and counts[2] < N:
            counts[2] += 1
            filtered_dataset.append((input, label))
        if label.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0] and counts[3] < N:
            counts[3] += 1
            filtered_dataset.append((input, label))
        if label.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0] and counts[4] < N:
            counts[4] += 1
            filtered_dataset.append((input, label))

    filtered_dataloader = DataLoader(filtered_dataset, sampler=None, batch_size=dataloader.batch_size,
                                     shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return filtered_dataloader

def filter_dataloader_sani_e_malati(dataloader):
    filtered_dataset = []
    counts = [0, 0]
    N = 400
    for input, label in dataloader.dataset:
        if label.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0] and counts[0] < N:
            counts[0] += 1
            filtered_dataset.append((input, label))
        if label.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0] and counts[1] < N:
            counts[1] += 1
            filtered_dataset.append((input, label))

    filtered_dataloader = DataLoader(filtered_dataset, sampler=None, batch_size=dataloader.batch_size,
                                     shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return filtered_dataloader

class Trainer:
    def __init__(self, single_prompt, prompts, class_names, loss_name, lr, device, writer):
        self.pos_mean_counter = 0
        self.neg_mean_counter = 0
        self.n_reset = 0
        self.n_updated = 0
        self.image_adapter_copy = None
        self.image_adapter_copy = None

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
                raise Exception
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

        print("image adapter", self.image_adapter)  # print summary
        print("text adapter", self.text_adapter)  # print summary
        if len(params) > 0:
            if OPTIM == "adam":
                print("Creating Adam optimizer...")
                self.optimizer = optim.Adam(params, lr=lr)
            elif OPTIM == "sgd":
                print("Creating SGD optimizer...")
                self.optimizer = optim.SGD(params, lr=lr)
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

    @torch.no_grad()
    def my_scatter_plt(self, x_axis, y_axis, metric, epoch, mode):
        fig = plt.figure()
        plt.scatter(x_axis, y_axis)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.title('Class ' + metric)
        self.writer.add_figure(mode + ' Class-metric/Class ' + metric, fig, epoch)
        plt.clf()
        plt.close()

    @staticmethod
    def _preprocessing(chex_competition, xrays_position, batch_size):
        # xxx CHEX COMPETITION
        if chex_competition:
            print("*** CHEX COMPETITION ***")
            class_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
            chex_str = "-chex"
        # xxx NOT CHEX competition
        else:
            raise Exception
            print("*** NO chex competition ***")
            # class_names = ["Pleural Effusion", "Pneumothorax", "Atelectasis", "Pneumonia", "Consolidation"]
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

        print("TrainBS:", batch_size, "Val/Test Batch size default set to 1024")
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=None, batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4, pin_memory=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, sampler=None, batch_size=1024,
                                                 shuffle=True,
                                                 num_workers=4, pin_memory=True, drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, sampler=None, batch_size=1024,
                                                  shuffle=True,
                                                  num_workers=4, pin_memory=True, drop_last=False)

        print("pre plot tsne loader split", len(train_loader.dataset))
        plot_tsne_loader_multiclass = filter_dataloader_multiclass(train_loader)
        plot_tsne_loader_sani_malati = filter_dataloader_sani_e_malati(train_loader)
        print("post plot tsne-multiclass loader", len(plot_tsne_loader_multiclass.dataset))
        print("post plot tsne-sani-malati loader", len(plot_tsne_loader_sani_malati.dataset))

        return class_names, chex_str, train_loader, val_loader, test_loader, [plot_tsne_loader_multiclass, plot_tsne_loader_sani_malati]

    @staticmethod
    def preprocessing(chex_competition, xrays_position, single_prompt, batch_size, lr, epochs, loss_name):

        class_names, chex_str, train_loader, val_loader, test_loader, plot_tsne_array = Trainer._preprocessing(
            chex_competition,
            xrays_position,
            batch_size)

        folder_name = "zero-and-joint-bounds-new-prompts"
        folder_name = "new-prompts-only-pos"
        folder_name = "INTATA"
        folder_name = "NUOVI_RISULTATI/vera-ultima-sperimentazione-zero-and-joint"
        folder_name = "NUOVI_RISULTATI-2/zero-and-joint"
        folder_name = "NUOVI_RISULTATI-3/zero-and-joint"
        if single_prompt:
            str_basic = "-single-prompt"
            prompts = basic_create_prompts(class_names)
        else:
            str_basic = "-mean-prompt"
            if MAX_EMB:
                str_basic = "-MAX-prompt"
            prompts = create_prompts(class_names, NEW_PROMPTS, TRAIN_LOGIT_DIFF)
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
                    suffix += "-only-text-adapter"

            w_path = "./" + folder_name + "/joint-train-loss-" + str(loss_name) + "-opt-" + OPTIM + "-lr-" + str(
                lr) + "-bs" + str(
                batch_size) + "-ep" + str(
                epochs) + chex_str + str_basic + "-" + str(xrays_position) + suffix
        if epochs == 0:
            if SHARED and IMAGE_MODEL and TEXT_MODEL:
                suffix = "-SHARED-adapter-" + MODEL_USED
            elif not SHARED and not IMAGE_MODEL and not TEXT_MODEL:
                suffix = "-no-head"
            else:
                raise Exception
            print("Attenzione! Zero-shot evaluation!")
            w_path = "./" + folder_name + "/zero-shot-model" + chex_str + str_basic + "-" + str(
                xrays_position) + suffix
        # w_path = "./joint-training/rapid_check"
        # w_path = w_path + "-There-is-There-is-no"
        if NEW_PROMPTS:
            w_path = w_path + "-NEW-PROMPTS"
        # if UNDER_SAMPLE:
        #     w_path = w_path + "-UNDER-SAMPLE"

        if TRAIN_LOGIT_DIFF:
            w_path = w_path + "-TRAIN-logit-DIFF"
        else:
            w_path = w_path + "-TRAIN-logit-POS"
        if PRED_LOGIT_DIFF:
            w_path = w_path + "-PRED-logit-DIFF"
        else:
            w_path = w_path + "-PRED-logit-POS"
        # w_path = w_path + "-2-PROMPT-UGUALI-PER-TUTTI"
        # w_path = w_path + "-debug-2"
        # w_path = w_path + "-CHEX-PROMPT"
        print("writer path:", w_path)
        writer = SummaryWriter(w_path)
        if True:
            for c in class_names:
                print(c, "positive", prompts[c]["positive"])
                print(c, "negative", prompts[c]["negative"])

        return writer, class_names, train_loader, val_loader, test_loader, prompts, plot_tsne_array

    @staticmethod
    def preprocessing_class_incremental(chex_competition, xrays_position, basic_prompts, batch_size, lr,
                                        epochs, loss_name, mode, CONTINUAL_LEARNING=None, ratio=None,
                                        threshold=None, threshold_scheduling=False, adder=0.01, MORE_LABELS=False):

        if CONTINUAL_LEARNING is not None:
            print("**** Gradient Clipping ****")
            print("--->", CONTINUAL_LEARNING)
        else:
            print("**** NO Gradient Clipping ****")

        class_names, chex_str, train_loader, val_loader, test_loader, plot_tsne_array = Trainer._preprocessing(
            chex_competition,
            xrays_position,
            batch_size)
        folder_name = "./ultimo-only-class-pos"
        folder_name = "./NUOVI_RISULTATI/" + mode
        if MORE_LABELS:
            folder_name += "-more-labels"
        if mode == "class-pos-neg":
            train_loader = Trainer.concat_to_tensor_dataloader(train_loader)
            train_loader = Trainer.split_dataloader_data_incremental(train_loader, 5)
        elif mode == "class-pos":
            train_loader = Trainer.concat_to_tensor_dataloader(train_loader)
            train_loader = Trainer.split_dataloader_by_label(train_loader, batch_size=batch_size)
            print()
        else:
            raise Exception

        if False:
            for i in range(5):
                print("Task", i, len(train_loader[i].dataset))
                Trainer.count_positive_labels(train_loader[i])
                print()

            print("Val", i, len(val_loader.dataset))
            Trainer.count_positive_labels(val_loader)
            print()

            print("Test", i, len(test_loader.dataset))
            Trainer.count_positive_labels(test_loader)
            print()

        thre_str = ""
        if threshold_scheduling and CONTINUAL_LEARNING is not None:
            thre_str = "-th-scheduled-" + str(adder)
        cl_str = ""
        if CONTINUAL_LEARNING is not None and ratio:
            cl_str = "-" + str(CONTINUAL_LEARNING) + "-ratio-" + str(threshold)
            mode = "gradient-clipping-" + mode
        else:
            mode = "fine-tuning-" + mode
        # mode = "fine-tuning-" + mode
        if basic_prompts:
            str_basic = "-single-prompt"
            prompts = basic_create_prompts(class_names)
        else:
            str_basic = "-mean-prompt"
            if MAX_EMB:
                str_basic = "-MAX-prompt"
            prompts = create_prompts(class_names, NEW_PROMPTS, TRAIN_LOGIT_DIFF)
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
            # w_path = "./only-" + mode + "/" + mode + "-loss-" + str(loss_name) + "-opt-" + OPTIM + "-lr-" + str(
            #     lr) + "-bs" + str(
            #     batch_size) + "-ep" + str(
            #     epochs) + chex_str + str_basic + "-" + str(xrays_position) + suffix + cl_str
            w_path = folder_name + "/" + mode + "-loss-" + str(loss_name) + "-opt-" + OPTIM + "-lr-" + str(
                lr) + "-bs" + str(
                batch_size) + "-ep" + str(
                epochs) + chex_str + str_basic + "-" + str(xrays_position) + suffix + cl_str + thre_str

        if epochs == 0:
            raise Exception
        # w_path = "./joint-training/rapid_check"
        # w_path = w_path + "-DEBUG"
        if MORE_LABELS:
            w_path += "-MORE-LABELS"
        if NEW_PROMPTS:
            w_path = w_path + "-NEW-PROMPTS"
        # if UNDER_SAMPLE:
        #     w_path = w_path + "-UNDER-SAMPLE"

        if TRAIN_LOGIT_DIFF:
            w_path = w_path + "-TRAIN-logit-DIFF"
        else:
            w_path = w_path + "-TRAIN-logit-POS"
        if PRED_LOGIT_DIFF:
            w_path = w_path + "-PRED-logit-DIFF"
        else:
            w_path = w_path + "-PRED-logit-POS"

        w_path = w_path + "-DD"
        print("writer path:", w_path)
        writer = SummaryWriter(w_path)

        return writer, class_names, train_loader, val_loader, test_loader, prompts, plot_tsne_array

    @staticmethod  # xxx prep for data-incremental
    def preprocessing_data_incremental(chex_competition, xrays_position, basic_prompts, batch_size, lr,
                                       parts, epochs, loss_name, mode, CONTINUAL_LEARNING=None, ratio=None,
                                       threshold=None, threshold_scheduling=False, adder=0.01):

        folder_name = "NUOVI_RISULTATI/data-incremental-" + str(parts) + "-parts"
        if CONTINUAL_LEARNING is not None:
            print("**** Gradient Clipping ****")
            print("--->", CONTINUAL_LEARNING)
        else:
            print("**** NO Gradient Clipping ****")

        class_names, chex_str, train_loader, val_loader, test_loader, plot_tsne_array = Trainer._preprocessing(
            chex_competition,
            xrays_position,
            batch_size)

        if mode == "data-inc":
            print("number of parts:", parts)
            train_loader = Trainer.split_dataloader_data_incremental(train_loader, parts)
            # Trainer.print_dataloader_stats(train_loader)  # to do unmute
        else:
            raise Exception

        thre_str = ""
        if CONTINUAL_LEARNING is not None and threshold_scheduling:
            thre_str = "-th-scheduled-" + str(adder)
        cl_str = ""
        if CONTINUAL_LEARNING is not None and ratio:
            cl_str = "-" + str(CONTINUAL_LEARNING) + "-ratio-" + str(threshold)
            mode = "gradient-clipping-" + mode
        else:
            mode = "fine-tuning-" + mode
        # mode = "fine-tuning-" + mode
        if basic_prompts:
            str_basic = "-single-prompt"
            prompts = basic_create_prompts(class_names)
        else:
            str_basic = "-mean-prompt"
            if MAX_EMB:
                str_basic = "-MAX-prompt"
            prompts = create_prompts(class_names, NEW_PROMPTS, TRAIN_LOGIT_DIFF)
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
            # w_path = "./only-" + mode + "/" + mode + "-loss-" + str(loss_name) + "-opt-" + OPTIM + "-lr-" + str(
            #     lr) + "-bs" + str(
            #     batch_size) + "-ep" + str(
            #     epochs) + chex_str + str_basic + "-" + str(xrays_position) + suffix + cl_str
            w_path = folder_name + "/" + mode + "-loss-" + str(
                loss_name) + "-opt-" + OPTIM + "-lr-" + str(
                lr) + "-bs" + str(
                batch_size) + "-ep" + str(
                epochs) + "-parts" + str(
                parts) + chex_str + str_basic + "-" + str(xrays_position) + suffix + cl_str + thre_str

        if epochs == 0:
            raise Exception
        # w_path = "./joint-training/rapid_check"
        # w_path = w_path + "-DEBUG"
        if NEW_PROMPTS:
            w_path = w_path + "-NEW-PROMPTS"
        # if UNDER_SAMPLE:
        #     w_path = w_path + "-UNDER-SAMPLE"

        if TRAIN_LOGIT_DIFF:
            w_path = w_path + "-TRAIN-logit-DIFF"
        else:
            w_path = w_path + "-TRAIN-logit-POS"
        if PRED_LOGIT_DIFF:
            w_path = w_path + "-PRED-logit-DIFF"
        else:
            w_path = w_path + "-PRED-logit-POS"

        w_path = w_path + "-DD"
        print("writer path:", w_path)
        writer = SummaryWriter(w_path)

        return writer, class_names, train_loader, val_loader, test_loader, prompts, plot_tsne_array

    # xxx works for normal and for data-incremental
    def train(self, train_loader, criterion, epoch, CONTINUAL_LEARNING=None, threshold=None,
              scheduler=None, part=None, epochs=None, actual_task=None):
        batch_idx = 0
        if IMAGE_MODEL:
            self.image_adapter.train()
        if TEXT_MODEL:
            self.text_adapter.train()
        if part is None:
            str_part = ""
        else:
            str_part = " part-" + str(part)
        for embs, labels in tqdm(train_loader, desc="Fine-tuning on chexpert," + str_part + " Epoch " + str(epoch)):
            if CONTINUAL_LEARNING == "myCL":
                if actual_task > 1:
                    self.model_copy()
            self.optimizer.zero_grad()
            batch_idx += 1
            embs = embs.to(self.device)
            labels = labels.to(self.device)
            if IMAGE_MODEL:
                new_embs = self.image_adapter(embs)
            else:
                new_embs = embs
            # new_embs = F.normalize(new_embs, dim=-1)  # todo remove

            if self.loss_name == "standard":
                logits = torch.empty(labels.shape[0], 5).to(self.device)
            else:
                raise Exception

            i = -1
            for label_name in self.class_names:
                i += 1
                if TRAIN_LOGIT_DIFF:
                    pos_prompt = self.prompts[label_name]["positive"]
                    neg_prompt = self.prompts[label_name]["negative"]
                else:
                    pos_prompt = self.prompts[label_name]["positive"]
                    neg_prompt = self.prompts[label_name]["positive"]  # xxx trick per non riscrivere il codice

                pos_prompt_embedding, neg_prompt_embedding = self.bert_forward_mean(pos_prompt, neg_prompt,
                                                                                    use_grad=True)

                pos_similarities = self.myCosineSimilarity(new_embs, pos_prompt_embedding, use_grad=True, train=True,
                                                           pos=True)
                neg_similarities = self.myCosineSimilarity(new_embs, neg_prompt_embedding, use_grad=True, train=True,
                                                           pos=False)

                if TRAIN_LOGIT_DIFF:
                    logits[:, i] = pos_similarities.flatten() - neg_similarities.flatten()
                else:
                    logits[:, i] = pos_similarities.flatten()

            if self.change_labels:
                labels = change_values(labels)

            loss = criterion(logits, labels)
            loss.backward()

            self.optimizer.step()

            if part is None:
                iteration = (epoch - 1) * len(train_loader) + batch_idx
            else:
                iteration = (part - 1) * epochs * len(train_loader) + (epoch - 1) * len(train_loader) + batch_idx

            if CONTINUAL_LEARNING == "myCL":
                if actual_task > 1:
                    self.myIncremental(threshold, iteration)

            if self.writer is not None:
                self.writer.add_scalar('train/Loss', loss.item(), iteration)
            if scheduler is not None:
                scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/LR', current_lr, iteration)

        if CONTINUAL_LEARNING == "myCL":
            if actual_task > 1:
                self.myIncremental_save_log(iteration)

    # xxx works for class-incremental one class and both class
    def train_class_incremental(self, train_loader, criterion, epoch, CONTINUAL_LEARNING=None, threshold=None,
                                current_task=None,
                                last_batch=0, actual_task=None):
        # xxx ONE EPOCH TRAIN
        batch_idx = last_batch
        if IMAGE_MODEL:
            self.image_adapter.train()
        if TEXT_MODEL:
            self.text_adapter.train()
        for embs, labels in tqdm(train_loader,
                                 desc="Fine-tuning on task " + str(current_task) + ", Epoch " + str(epoch)):
            if CONTINUAL_LEARNING == "myCL" and actual_task > 1:
                self.model_copy()

            self.optimizer.zero_grad()
            batch_idx += 1
            embs = embs.to(self.device)
            labels = labels.to(self.device)
            labels = labels[:, current_task]
            if IMAGE_MODEL:
                new_embs = self.image_adapter(embs)
            else:
                new_embs = embs
            # new_embs = F.normalize(new_embs, dim=-1)  # todo remove

            if self.loss_name == "standard":
                logits = torch.zeros(labels.shape[0]).to(self.device)
            else:
                raise Exception  # logits = torch.empty(labels.shape[0], 2).to(self.device)

            # for label_name in self.class_names:
            label_name = self.class_names[current_task]

            if TRAIN_LOGIT_DIFF:
                pos_prompt = self.prompts[label_name]["positive"]
                neg_prompt = self.prompts[label_name]["negative"]
            else:
                pos_prompt = self.prompts[label_name]["positive"]
                neg_prompt = self.prompts[label_name]["positive"]  # xxx trick per non riscrivere il codice

            pos_prompt_embedding, neg_prompt_embedding = self.bert_forward_mean(pos_prompt, neg_prompt, use_grad=True)

            # Calculate the similarities between the image and the positive and negative prompts
            # pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
            # neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)
            pos_similarities = self.myCosineSimilarity(new_embs, pos_prompt_embedding, use_grad=True, train=True,
                                                       pos=True)
            neg_similarities = self.myCosineSimilarity(new_embs, neg_prompt_embedding, use_grad=True, train=True,
                                                       pos=False)

            if TRAIN_LOGIT_DIFF:
                logits = pos_similarities.flatten() - neg_similarities.flatten()
            else:
                logits = pos_similarities.flatten()
            # Compute loss and backpropagate
            # loss figa con labels -2, 2 che ipoteticamnete spara pos a 1 neg a -1 e viceversa
            if self.change_labels:
                labels = change_values(labels)
            # loss = criterion(predicted_labels, labels)

            loss = criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            iteration = batch_idx
            if CONTINUAL_LEARNING == "myCL" and actual_task > 1:
                self.myIncremental(threshold, iteration)

            if self.writer is not None:
                self.writer.add_scalar('train/Loss', loss.item(), iteration)
        if CONTINUAL_LEARNING == "myCL" and actual_task > 1:
            self.myIncremental_save_log(iteration)
        return iteration

    def train_class_more_labels_incremental(self, train_loader, criterion, epoch, CONTINUAL_LEARNING=None,
                                            threshold=None,
                                            current_task=None,
                                            last_batch=0, actual_task=None):
        # ONE EPOCH TRAIN
        batch_idx = last_batch
        if IMAGE_MODEL:
            self.image_adapter.train()
        if TEXT_MODEL:
            self.text_adapter.train()
        for embs, labels in tqdm(train_loader,
                                 desc="Fine-tuning on task " + str(current_task) + ", Epoch " + str(epoch)):
            if CONTINUAL_LEARNING == "myCL" and actual_task > 1:
                self.model_copy()

            self.optimizer.zero_grad()
            batch_idx += 1
            embs = embs.to(self.device)
            labels = labels.to(self.device)
            labels = labels[:, :current_task + 1]
            if IMAGE_MODEL:
                new_embs = self.image_adapter(embs)
            else:
                new_embs = embs
            # new_embs = F.normalize(new_embs, dim=-1)  # todo remove

            if self.loss_name == "standard":
                logits = torch.empty(labels.shape[0], current_task + 1).to(self.device)
            else:
                raise Exception  # logits = torch.empty(labels.shape[0], 2).to(self.device)

            i = -1
            for label_name in self.class_names[:current_task + 1]:
                i += 1

                if TRAIN_LOGIT_DIFF:
                    pos_prompt = self.prompts[label_name]["positive"]
                    neg_prompt = self.prompts[label_name]["negative"]
                else:
                    pos_prompt = self.prompts[label_name]["positive"]
                    neg_prompt = self.prompts[label_name]["positive"]  # xxx trick per non riscrivere il codice

                pos_prompt_embedding, neg_prompt_embedding = self.bert_forward_mean(pos_prompt, neg_prompt,
                                                                                    use_grad=True)

                pos_similarities = self.myCosineSimilarity(new_embs, pos_prompt_embedding, use_grad=True, train=True,
                                                           pos=True)
                neg_similarities = self.myCosineSimilarity(new_embs, neg_prompt_embedding, use_grad=True, train=True,
                                                           pos=False)
                if TRAIN_LOGIT_DIFF:
                    logits[:, i] = pos_similarities.flatten() - neg_similarities.flatten()
                else:
                    logits[:, i] = pos_similarities.flatten()
                # Compute loss and backpropagate
            # loss figa con labels -2, 2 che ipoteticamnete spara pos a 1 neg a -1 e viceversa
            if self.change_labels:
                labels = change_values(labels)
            # loss = criterion(predicted_labels, labels)

            loss = criterion(logits, labels)
            loss.backward()

            self.optimizer.step()

            iteration = batch_idx
            if CONTINUAL_LEARNING == "myCL" and actual_task > 1:
                self.myIncremental(threshold, iteration)

            if self.writer is not None:
                self.writer.add_scalar('train/Loss', loss.item(), iteration)

        if CONTINUAL_LEARNING == "myCL" and actual_task > 1:
            self.myIncremental_save_log(iteration)

        return iteration

    @torch.no_grad()
    def myIncremental_save_log(self, iteration):
        print()
        print("number of resets:", self.n_reset.item(), "number of updates:", self.n_updated.item(),
              "percentage resets",
              self.n_reset.item() / (self.n_reset.item() + self.n_updated.item()))
        self.writer.add_scalar("monitor-resets/resets", self.n_reset.item(), iteration)
        self.writer.add_scalar("monitor-resets/updates", self.n_updated.item(), iteration)
        self.writer.add_scalar("monitor-resets/percentage resets",
                               self.n_reset.item() / (self.n_reset.item() + self.n_updated.item()),
                               iteration)
        self.n_reset = 0
        self.n_updated = 0

    @torch.no_grad()
    def val(self, val_loader, criterion, epoch, epochs, mode="joint", tasks_order=None):
        '''
        mode = "joint" or "class-pos-neg"/ "class-pos"
        if "class" we need to set "tasks_order"
        '''
        batch_idx = 0
        y_true = []
        y_pred = []
        y_score = []
        if IMAGE_MODEL:
            self.image_adapter.eval()
        if TEXT_MODEL:
            self.text_adapter.eval()
        with torch.no_grad():
            for embs, labels in tqdm(val_loader, desc="Validating on chexpert mode: " + mode + ", Epoch " + str(epoch)):
                batch_idx += 1
                embs = embs.to(self.device)
                labels = labels.to(self.device)
                if IMAGE_MODEL:
                    new_embs = self.image_adapter(embs)
                else:
                    new_embs = embs
                # new_embs = F.normalize(new_embs, dim=-1)  # todo remove

                predicted_labels = torch.zeros(labels.shape[0], 5).to(self.device)
                tmp_score = torch.zeros(labels.shape[0], 5).to(self.device)

                if self.loss_name == "standard":
                    logits = torch.empty(labels.shape[0], 5).to(self.device)
                else:
                    raise Exception

                i = -1
                for label_name in self.class_names:
                    i += 1

                    if TRAIN_LOGIT_DIFF:
                        pos_prompt = self.prompts[label_name]["positive"]
                        neg_prompt = self.prompts[label_name]["negative"]
                    else:
                        pos_prompt = self.prompts[label_name]["positive"]
                        neg_prompt = self.prompts[label_name]["positive"]  # xxx trick per non riscrivere il codice

                    pos_prompt_embedding, neg_prompt_embedding = self.bert_forward_mean(pos_prompt, neg_prompt,
                                                                                        use_grad=False)

                    # pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
                    # neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)
                    pos_similarities = self.myCosineSimilarity(new_embs, pos_prompt_embedding, use_grad=False)
                    neg_similarities = self.myCosineSimilarity(new_embs, neg_prompt_embedding, use_grad=False)

                    if not PRED_LOGIT_DIFF:
                        tmp_score[:, i] = (pos_similarities.flatten() + 1) / 2
                    if PRED_LOGIT_DIFF:
                        tmp_score[:, i] = (pos_similarities.flatten() - neg_similarities.flatten() + 2) / 4

                    if TRAIN_LOGIT_DIFF:
                        logits[:, i] = pos_similarities.flatten() - neg_similarities.flatten()
                    else:
                        logits[:, i] = pos_similarities.flatten()

                    pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
                    neg_similarities = neg_similarities.reshape(-1, 1)
                    predicted_labels[:, i] = torch.argmax(torch.cat([neg_similarities, pos_similarities], dim=1),
                                                          dim=1)

                # Compute loss and backpropagate
                if self.change_labels:
                    tmp = labels
                    labels = change_values(labels)

                loss = criterion(logits, labels)

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
                y_score.append(tmp_score.cpu().numpy())

                # Concatenate the true and predicted labels
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_score = np.concatenate(y_score)

        self.evaluate_model(y_true, y_pred, y_score, mode, epoch, "val", epochs, tasks_order)

    @torch.no_grad()
    def evaluate_model(self, y_true, y_pred, y_score, mode, epoch, val_test, epochs, tasks_order):
        # Calculate the metrics
        accuracy = accuracy_score(y_true, y_pred)  # OK
        f1_macro = f1_score(y_true, y_pred, average="macro")  # OK
        f1_weighted = f1_score(y_true, y_pred, average="weighted")  # OK
        auroc_macro = roc_auc_score(y_true, y_score, average="macro", multi_class="ovr")
        auroc_weighted = roc_auc_score(y_true, y_score, average="weighted", multi_class="ovr")
        precision = precision_score(y_true, y_pred, average="weighted")  # OK
        recall = recall_score(y_true, y_pred, average="weighted")  # OK

        for i in range(5):
            fpr, tpr, thresholds = roc_curve(y_true[:, i], y_score[:, i])  # GIGAFIX
            auc = roc_auc_score(y_true[:, i], y_score[:, i])
            fig = plt.figure()
            plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc))
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Class ' + str(i))
            plt.legend(loc="lower right")
            self.writer.add_figure(val_test + " ROC Curve/Curve for Class " + str(i), fig, epoch)

            precision_i, recall_i, thresholds = precision_recall_curve(y_true[:, i], y_score[:, i])  # GIGAFIX
            ap = average_precision_score(y_true[:, i], y_score[:, i])
            fig = plt.figure()
            plt.plot(recall_i, precision_i, label='AP = {:.3f}'.format(ap))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve for Class ' + str(i))
            plt.legend(loc="lower left")
            self.writer.add_figure(val_test + " Precision-Recall Curve/Curve for Class " + str(i), fig, epoch)

        if self.writer is not None:
            self.writer.add_scalar(val_test + "/Accuracy", accuracy, epoch)
            self.writer.add_scalar(val_test + "/F1-macro score", f1_macro, epoch)
            self.writer.add_scalar(val_test + "/F1-weighted score", f1_weighted, epoch)
            self.writer.add_scalar(val_test + "/AUROC-macro", auroc_macro, epoch)
            self.writer.add_scalar(val_test + "/AUROC-weighted", auroc_weighted, epoch)

            if val_test == "val":
                self.val_f1_heat_map, self.val_auroc_heat_map = self._aux_evaluate_model(y_true, y_pred, y_score, mode,
                                                                                         epoch, val_test, epochs,
                                                                                         tasks_order,
                                                                                         self.val_f1_heat_map,
                                                                                         self.val_auroc_heat_map)
            elif val_test == "test":
                self.test_f1_heat_map, self.test_auroc_heat_map = self._aux_evaluate_model(y_true, y_pred, y_score,
                                                                                           mode, epoch, val_test,
                                                                                           epochs, tasks_order,
                                                                                           self.test_f1_heat_map,
                                                                                           self.test_auroc_heat_map)
            else:
                raise Exception

            x = [1, 2, 3, 4, 5]
            acc_y = []
            prec_y = []
            rec_y = []
            for i in range(5):
                acc_y.append(accuracy_score(y_true[:, i], y_pred[:, i]))  # OK
                prec_y.append(precision_score(y_true[:, i], y_pred[:, i]))  # OK
                rec_y.append(recall_score(y_true[:, i], y_pred[:, i]))  # OK
            self.my_scatter_plt(x_axis=x, y_axis=acc_y, metric="Accuracy", epoch=epoch, mode=val_test)
            self.my_scatter_plt(x_axis=x, y_axis=prec_y, metric="Precision", epoch=epoch, mode=val_test)
            self.my_scatter_plt(x_axis=x, y_axis=rec_y, metric="Recall", epoch=epoch, mode=val_test)

    @torch.no_grad()
    def _aux_evaluate_model(self, y_true, y_pred, y_score, mode, epoch, val_test, epochs, tasks_order, F1_MAP,
                            AUROC_MAP):
        tmp_f1 = torch.zeros(1, 5)
        tmp_auroc = torch.zeros(1, 5)
        for i in range(5):
            tmp_f1[0, i] = f1_score(y_true[:, i], y_pred[:, i])  # OK
            tmp_auroc[0, i] = roc_auc_score(y_true[:, i], y_score[:, i])
        F1_MAP = torch.cat([F1_MAP, tmp_f1], dim=0)
        AUROC_MAP = torch.cat([AUROC_MAP, tmp_auroc], dim=0)
        if epoch == epochs and (mode == "joint" or mode == "zero" or mode == "data-inc"):
            F1_MAP = numpy.array(F1_MAP)
            fig, ax = plt.subplots()
            im, cbar = heatmap(F1_MAP, [i for i in range(1, epochs + 1)],
                               [self.class_names[i] for i in range(0, 5)], ax=ax,
                               cmap="YlGn", cbarlabel="F1 score", metric="F1")
            texts = annotate_heatmap(im, valfmt="{x:.2f}")
            fig.tight_layout()
            # plt.show()
            self.writer.add_figure(val_test + "/joint train/F1 score Heatmap", fig)

            AUROC_MAP = numpy.array(AUROC_MAP)
            fig, ax = plt.subplots()
            im, cbar = heatmap(AUROC_MAP, [i for i in range(1, epochs + 1)],
                               [self.class_names[i] for i in range(0, 5)], ax=ax,
                               cmap="YlGn", cbarlabel="AUROC score", metric="AUROC")
            texts = annotate_heatmap(im, valfmt="{x:.2f}")
            fig.tight_layout()
            # plt.show()
            self.writer.add_figure(val_test + "/joint train/AUROC score Heatmap", fig)

        if epoch == 5 and (mode == "class-pos-neg" or mode == "class-pos"):
            F1_MAP = numpy.array(F1_MAP)
            fig, ax = plt.subplots()
            im, cbar = heatmap(F1_MAP, [self.class_names[i] for i in tasks_order],
                               [self.class_names[i] for i in tasks_order], ax=ax,
                               cmap="YlGn", cbarlabel="F1 score", metric="F1")
            texts = annotate_heatmap(im, valfmt="{x:.2f}")
            fig.tight_layout()
            # plt.show()
            self.writer.add_figure(val_test + "/" + mode + ' incremental/F1 score Heatmap', fig)

            AUROC_MAP = numpy.array(AUROC_MAP)
            fig, ax = plt.subplots()
            im, cbar = heatmap(AUROC_MAP, [self.class_names[i] for i in tasks_order],
                               [self.class_names[i] for i in tasks_order], ax=ax,
                               cmap="YlGn", cbarlabel="AUROC score", metric="AUROC")
            texts = annotate_heatmap(im, valfmt="{x:.2f}")
            fig.tight_layout()
            # plt.show()
            self.writer.add_figure(val_test + "/" + mode + ' incremental/AUROC score Heatmap', fig)

        return F1_MAP, AUROC_MAP

    @torch.no_grad()
    def test(self, test_loader, criterion, epoch, epochs, mode="joint", tasks_order=None, plot_tsne_array=None):
        '''
        mode = "joint" or "class-pos-neg"/ "class-pos"
        if "class" we need to set "tasks_order"
        '''
        y_true = []
        y_pred = []
        y_score = []
        if IMAGE_MODEL:
            self.image_adapter.eval()
        if TEXT_MODEL:
            self.text_adapter.eval()
        with torch.no_grad():
            for embs, labels in tqdm(test_loader, desc="Testing on chexpert mode: " + mode):
                # image0_to_plt = embs[0]
                # image0_to_plt = image0_to_plt.permute(1, 2, 0)
                # # Plot the RGB tensor
                # plt.imshow(image0_to_plt)
                # plt.show()
                embs = embs.to(self.device)
                labels = labels.to(self.device)
                if IMAGE_MODEL:
                    new_embs = self.image_adapter(embs)
                else:
                    new_embs = embs
                # new_embs = F.normalize(new_embs, dim=-1)  # todo remove

                predicted_labels = torch.zeros(labels.shape[0], 5).to(self.device)
                tmp_score = torch.zeros(labels.shape[0], 5).to(self.device)

                i = -1
                for label_name in self.class_names:
                    i += 1
                    # Get the positive and negative prompts for the label
                    if TRAIN_LOGIT_DIFF:
                        pos_prompt = self.prompts[label_name]["positive"]
                        neg_prompt = self.prompts[label_name]["negative"]
                    else:
                        pos_prompt = self.prompts[label_name]["positive"]
                        neg_prompt = self.prompts[label_name]["positive"]  # xxx trick per non riscrivere il codice

                    pos_prompt_embedding, neg_prompt_embedding = self.bert_forward_mean(pos_prompt, neg_prompt,
                                                                                        use_grad=False)

                    # Calculate the similarities between the image and the positive and negative prompts
                    # pos_similarities = torch.matmul(new_embs, pos_prompt_embedding.T)
                    # neg_similarities = torch.matmul(new_embs, neg_prompt_embedding.T)
                    pos_similarities = self.myCosineSimilarity(new_embs, pos_prompt_embedding, use_grad=False)
                    neg_similarities = self.myCosineSimilarity(new_embs, neg_prompt_embedding, use_grad=False)

                    if not PRED_LOGIT_DIFF:
                        tmp_score[:, i] = (pos_similarities.flatten() + 1) / 2
                    if PRED_LOGIT_DIFF:
                        tmp_score[:, i] = (pos_similarities.flatten() - neg_similarities.flatten() + 2) / 4

                    pos_similarities = pos_similarities.reshape(-1, 1)  # da (batch, a (batch, 1)
                    neg_similarities = neg_similarities.reshape(-1, 1)
                    predicted_labels[:, i] = torch.argmax(torch.cat([neg_similarities, pos_similarities], dim=1),
                                                          dim=1)

                # Convert the predicted labels to a numpy array
                predicted_labels_np = predicted_labels.cpu().numpy()

                # Append the true and predicted labels to the lists
                y_true.append(labels.cpu().numpy())
                y_pred.append(predicted_labels_np)
                y_score.append(tmp_score.cpu().numpy())

                # Concatenate the true and predicted labels
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_score = np.concatenate(y_score)

        self.evaluate_model(y_true, y_pred, y_score, mode, epoch, "test", epochs, tasks_order)

        if TRAIN_LOGIT_DIFF:
            self.plot_cosine_similarity_text_embs(epoch, epochs)
        else:
            self.plot_cosine_similarity_text_embs_only_pos_prompts(epoch, epochs)
        self.plot_new_text_embeddings(epoch, epochs)

        if plot_tsne_array is not None:
            self.plot_tsne_sani_malati(plot_tsne_array[1], epoch, epochs)
            self.plot_tsne_multiclass(plot_tsne_array[0], epoch, epochs)

    @torch.no_grad()
    def plot_tsne_multiclass(self, plot_tsne_loader, epoch, epochs):

        if IMAGE_MODEL:
            self.image_adapter.eval()
        if TEXT_MODEL:
            self.text_adapter.eval()

        abbrevviations = ["ATE", "CMG", "CONS", "EDE", "PLEF"]
        embeddings = []
        labels = []
        # tmp_colors = ['red', 'green', 'blue', 'yellow', 'purple']
        tmp_colors = ['#FEB24C', '#F03B20', '#74C476', '#238B8C', '#6A51A3']

        for embs, lbs in tqdm(plot_tsne_loader, desc="Tsne Plot multiclass"):
            embs = embs.to(self.device)
            lbs = lbs.to(self.device)
            if IMAGE_MODEL:
                new_embs = self.image_adapter(embs)
            else:
                new_embs = embs
            embeddings.append(new_embs)
            labels.append(lbs)
        embeddings = torch.cat(embeddings).cpu()
        labels = torch.cat(labels).cpu()

        colors = [tmp_colors[np.argmax(l)] for l in labels]

        abbrev_color_dict = {abbrev: color for abbrev, color in zip(abbrevviations, tmp_colors)}
        legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in abbrev_color_dict.values()]
        tsne = TSNE(n_components=2, metric='cosine', init="pca", learning_rate="auto", square_distances=True)
        fig = plt.figure()

        X_tsne = tsne.fit_transform(embeddings)
        # Plot t-SNE embedding with colored labels
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='viridis', s=25, alpha=0.7)
        # plt.colorbar()
        plt.legend(legend_patches, abbrev_color_dict.keys())
        plt.xticks([])
        plt.yticks([])
        plt.title('t-SNE Plot', fontsize=20)
        # plt.show()
        fig.canvas.draw()

        if epochs > 0 and epoch == 1:
            img_path = 'tsne-chex-5x1000.png'
            # load the PNG image
            image = read_image(img_path)
            self.writer.add_image('tsne-chexpert/t-SNE 5x1000', image, 0)

        if epoch == 0 and epochs == 0:
            self.writer.add_figure('tsne-chexpert/t-SNE 5x1000', fig, 0)
        if epochs > 0:
            self.writer.add_figure('tsne-chexpert/t-SNE 5x1000', fig, epoch)

    @torch.no_grad()
    def plot_tsne_sani_malati(self, plot_tsne_loader, epoch, epochs):

        if IMAGE_MODEL:
            self.image_adapter.eval()
        if TEXT_MODEL:
            self.text_adapter.eval()

        abbrevviations = ["NF", "DS"]
        embeddings = []
        labels = []
        # NF-0, DS-1
        tmp_colors = ['#F03B20', '#74C476']

        for embs, lbs in tqdm(plot_tsne_loader, desc="Tsne Plot sani e malati"):
            embs = embs.to(self.device)
            lbs = lbs.to(self.device)
            if IMAGE_MODEL:
                new_embs = self.image_adapter(embs)
            else:
                new_embs = embs
            embeddings.append(new_embs)
            labels.append(lbs)
        embeddings = torch.cat(embeddings).cpu()
        labels = torch.cat(labels).cpu()

        # labels[labels == [0.0, 0.0, 0.0, 0.0, 0.0]] = 0
        # labels[labels == [1.0, 1.0, 1.0, 1.0, 1.0]] = 1
        labels = torch.sum(labels, dim=1)/5
        colors = [tmp_colors[int(l)] for l in labels]

        abbrev_color_dict = {abbrev: color for abbrev, color in zip(abbrevviations, tmp_colors)}
        # print(abbrev_color_dict)
        legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in abbrev_color_dict.values()]
        tsne = TSNE(n_components=2, metric='cosine', init="pca", learning_rate="auto", square_distances=True)
        fig = plt.figure()

        X_tsne = tsne.fit_transform(embeddings)
        # Plot t-SNE embedding with colored labels
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, cmap='viridis', s=25, alpha=0.7)
        # plt.colorbar()
        plt.legend(legend_patches, abbrev_color_dict.keys())
        plt.xticks([])
        plt.yticks([])
        plt.title('t-SNE Plot', fontsize=20)
        # plt.show()
        fig.canvas.draw()
        if epochs > 0 and epoch == 1:
            img_path = 'tsne-chex-sani-malati.png'
            # load the PNG image
            image = read_image(img_path)
            self.writer.add_image('tsne-chexpert/t-SNE sani-malati', image, 0)

        if epoch == 0 and epochs == 0:
            self.writer.add_figure('tsne-chexpert/t-SNE sani-malati', fig, 0)
        if epochs > 0:
            self.writer.add_figure('tsne-chexpert/t-SNE sani-malati', fig, epoch)

    @staticmethod  # xxx for class-incremental one class with intersection
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

    @staticmethod  # xxx for data-incremental and class-incremental two class no intersection
    def split_dataloader_data_incremental(dataloader, n):
        """
        Splits a PyTorch DataLoader object with a ConcatDataset into N smaller DataLoader objects of equal size.
        """
        dataset = dataloader.dataset
        num_samples = len(dataset)
        subset_size = math.ceil(num_samples / n)
        subsets = [Subset(dataset, range(i * subset_size, min((i + 1) * subset_size, num_samples))) for i in range(n)]

        # dataloaders = [
        #     DataLoader(subset, batch_size=dataloader.batch_size, sampler=RandomSampler(subset), num_workers=1,
        #                pin_memory=True, drop_last=False, persistent_workers=True) for subset in subsets]
        dataloaders = [
            DataLoader(subset, batch_size=dataloader.batch_size, sampler=RandomSampler(subset), num_workers=1,
                       pin_memory=True, drop_last=False, persistent_workers=True) for subset in subsets]

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
        Prints statistics for each DataLoader in a list of PyTorch DataLoader objects
        """
        for i, dataloader in enumerate(dataloaders):
            print(f"Dataloader {i}:")
            print(f"Length: {len(dataloader.dataset)}")
            label_counts = [0] * 5
            for x, labels in dataloader:
                for j in range(5):
                    label_counts[j] += (labels[:, j] == 1).sum().item()
            print(f"Label counts: {label_counts}")

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

    @torch.no_grad()
    def plot_new_text_embeddings(self, epoch, epochs):
        abbrevviations = ["ATEL-pos", "ATEL-neg", "CMG-pos", "CMG-neg", "CONS-pos", "CONS-neg",
                          "EDE-pos", "EDE-neg", "PLEF-pos", "PLEF-neg"]
        embeddings = []
        shapes = ['o', 'v', 'o', 'v', 'o', 'v', 'o', 'v', 'o', 'v']
        class_groups = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}
        group_colors = ['r', 'g', 'b', 'c', 'm']
        colors = [group_colors[class_groups[i]] for i in range(10)]

        for label_name in self.class_names:
            if TRAIN_LOGIT_DIFF:
                pos_prompt = self.prompts[label_name]["positive"]
                neg_prompt = self.prompts[label_name]["negative"]
            else:
                pos_prompt = self.prompts[label_name]["positive"]
                neg_prompt = self.prompts[label_name]["positive"]  # xxx trick per non riscrivere il codice

            pos_prompt_embedding, neg_prompt_embedding = self.bert_forward_mean(pos_prompt, neg_prompt, use_grad=False,
                                                                                to_plot=True)

            embeddings.append(pos_prompt_embedding)
            embeddings.append(neg_prompt_embedding)

        embeddings = torch.stack(embeddings).cpu()

        # perform PCA on the embeddings to reduce them to 2 dimensions
        # run block of code and catch warnings

        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        # create new Figure object
        fig = plt.figure()

        # plot the reduced embeddings
        for i in range(10):
            plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], marker=shapes[i], c=colors[i],
                        label=f'class{i}')
        plt.title("PCA multiple-prompts")

        # create legend
        legend_categories = {'r': 'ATEL', 'g': 'CMG', 'b': 'CONS', 'c': 'EDE', 'm': 'PLEF'}
        legend_shapes = {'o': 'Positive', 'v': 'Negative'}
        handles = []
        for color, category in legend_categories.items():
            handles.append(
                plt.Line2D([0], [0], marker='o', color='w', label=category, markerfacecolor=color, markersize=10))
        for shape, label in legend_shapes.items():
            handles.append(
                plt.Line2D([0], [0], marker=shape, color='w', label=label, markerfacecolor='k', markersize=10))
        plt.legend(handles=handles)

        # convert plot to image and add it to SummaryWriter
        fig.canvas.draw()  # draw the figure
        # self.writer.add_figure('visual-embeddings/PCA text-embs', fig, epoch)
        if epochs > 0 and epoch == 1:
            img_path = 'pca_multiple_prompts.png'
            # load the PNG image
            image = read_image(img_path)
            # convert the image to a grid
            self.writer.add_image('visual-embeddings/PCA text-embs', image, 0)

        if epoch == 0 and epochs == 0:  # zero-shot recalculated
            self.writer.add_figure('visual-embeddings/PCA text-embs', fig, 0)
        if epochs > 0:
            self.writer.add_figure('visual-embeddings/PCA text-embs', fig, epoch)

        # perform t-SNE on the embeddings to reduce them to 2 dimensions
        with warnings.catch_warnings():
            # ignore all caught warnings
            warnings.filterwarnings("ignore")
            # execute code that will generate warnings
            tsne = TSNE(n_components=2, metric="cosine", init="pca", learning_rate="auto")
            reduced_embeddings = tsne.fit_transform(embeddings)
        # create new Figure object
        fig = plt.figure()

        # plot the reduced embeddings
        for i in range(10):
            plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], marker=shapes[i], c=colors[i],
                        label=f'class{i}')
        plt.title("TSNE multiple-prompts")

        # create legend
        legend_categories = {'r': 'ATEL', 'g': 'CMG', 'b': 'CONS', 'c': 'EDE', 'm': 'PLEF'}
        legend_shapes = {'o': 'Positive', 'v': 'Negative'}
        handles = []
        for color, category in legend_categories.items():
            handles.append(
                plt.Line2D([0], [0], marker='o', color='w', label=category, markerfacecolor=color, markersize=10))
        for shape, label in legend_shapes.items():
            handles.append(
                plt.Line2D([0], [0], marker=shape, color='w', label=label, markerfacecolor='k', markersize=10))
        plt.legend(handles=handles)

        # convert plot to image and add it to SummaryWriter
        fig.canvas.draw()  # draw the figure

        # self.writer.add_figure('visual-embeddings/t-SNE text-embs', fig, epoch)
        if epochs > 0 and epoch == 1:
            img_path = 'tsne_multiple_prompts.png'
            # load the PNG image
            image = read_image(img_path)
            # convert the image to a grid
            self.writer.add_image('visual-embeddings/t-SNE text-embs', image, 0)

        if epoch == 0 and epochs == 0:
            self.writer.add_figure('visual-embeddings/t-SNE text-embs', fig, 0)
        if epochs > 0:
            self.writer.add_figure('visual-embeddings/t-SNE text-embs', fig, epoch)

    @torch.no_grad()
    def plot_cosine_similarity_text_embs_only_pos_prompts(self, epoch, epochs):
        '''
        Atelectasis: ATEL
        Cardiomegaly: CMG
        Consolidation: CONS
        Edema: EDE
        Pleural Effusion: PLEF
        '''
        abbrevviations = ["ATEL-pos", "CMG-pos", "CONS-pos", "EDE-pos", "PLEF-pos"]

        cosine_similarity_heatmap = torch.zeros((5, 5))

        for i, label_name_i in enumerate(self.class_names):
            pos_prompt = self.prompts[label_name_i]["positive"]

            pos_emb_i, _ = self.bert_forward_mean(pos_prompt, pos_prompt, use_grad=False, to_plot=True)

            for j, label_name_j in enumerate(self.class_names):
                # print(j, label_name_j)
                pos_prompt = self.prompts[label_name_j]["positive"]

                pos_emb_j, _ = self.bert_forward_mean(pos_prompt, pos_prompt, use_grad=False, to_plot=True)

                # pos_similarities = torch.matmul(pos_emb_i, pos_emb_j.T)  # TODO check if this is correct
                pos_similarities = self.myCosineSimilarity(pos_emb_i, pos_emb_j, use_grad=False, to_plot=True)

                cosine_similarity_heatmap[i, j] = pos_similarities

        if self.basic_prompts:
            str_prompts = "-single-prompt"
        else:
            str_prompts = "-multiple-prompts"
        heat_map = numpy.array(cosine_similarity_heatmap)
        fig, ax = plt.subplots()
        im, cbar = heatmap(heat_map, [i for i in abbrevviations], [j for j in abbrevviations],
                           ax=ax,
                           cmap="YlGn", cbarlabel="Cosine similarity heatmap" + str_prompts, metric="COS")
        texts = annotate_heatmap(im, valfmt="{x:.2f}")
        fig.tight_layout()
        if epochs > 0 and epoch == 1:
            img_path = 'cosine_similarity_heat_map_fix.png'
            # load the PNG image
            image = read_image(img_path)
            # convert the image to a grid
            self.writer.add_image('visual-embeddings/cosine-similarity Heatmap text-embs', image, 0)

        if epoch == 0 and epochs == 0:
            self.writer.add_figure('visual-embeddings/cosine-similarity Heatmap text-embs', fig, 0)
        if epochs > 0:
            self.writer.add_figure('visual-embeddings/cosine-similarity Heatmap text-embs', fig, epoch)

    @torch.no_grad()
    def plot_cosine_similarity_text_embs(self, epoch, epochs):
        '''
        Atelectasis: ATEL
        Cardiomegaly: CMG
        Consolidation: CONS
        Edema: EDE
        Pleural Effusion: PLEF
        '''
        abbrevviations = ["ATEL-pos", "ATEL-neg", "CMG-pos", "CMG-neg", "CONS-pos", "CONS-neg",
                          "EDE-pos", "EDE-neg", "PLEF-pos", "PLEF-neg"]

        cosine_similarity_heatmap = torch.zeros((10, 10))

        for i, label_name_i in enumerate(self.class_names):
            pos_prompt = self.prompts[label_name_i]["positive"]
            neg_prompt = self.prompts[label_name_i]["negative"]

            pos_emb_i, neg_emb_i = self.bert_forward_mean(pos_prompt, neg_prompt, use_grad=False, to_plot=True)

            for j, label_name_j in enumerate(self.class_names):
                # print(j, label_name_j)
                pos_prompt = self.prompts[label_name_j]["positive"]
                neg_prompt = self.prompts[label_name_j]["negative"]

                pos_emb_j, neg_emb_j = self.bert_forward_mean(pos_prompt, neg_prompt, use_grad=False, to_plot=True)

                # pos_similarities_left = torch.matmul(pos_emb_i, pos_emb_j.T)  # TODO check if this is correct
                pos_similarities_left = self.myCosineSimilarity(pos_emb_i, pos_emb_j,
                                                                use_grad=False,
                                                                to_plot=True)  # TODO check if this is correct
                # if MAX_EMB:
                #     pos_similarities_left, inds = torch.max(pos_similarities_left, dim=0)
                cosine_similarity_heatmap[i * 2, j * 2] = pos_similarities_left

                # pos_similarities_right = torch.matmul(pos_emb_i, neg_emb_j.T)  # TODO check if this is correct
                pos_similarities_right = self.myCosineSimilarity(pos_emb_i, neg_emb_j,
                                                                 use_grad=False,
                                                                 to_plot=True)  # TODO check if this is correct
                # if MAX_EMB:
                #     pos_similarities_right, inds = torch.max(pos_similarities_right, dim=0)
                cosine_similarity_heatmap[i * 2, j * 2 + 1] = pos_similarities_right

                # neg_similarities_left = torch.matmul(neg_emb_i, pos_emb_j.T)  # TODO check if this is correct
                neg_similarities_left = self.myCosineSimilarity(neg_emb_i, pos_emb_j,
                                                                use_grad=False,
                                                                to_plot=True)  # TODO check if this is correct
                # if MAX_EMB:
                #     neg_similarities_left, inds = torch.max(neg_similarities_left, dim=0)
                cosine_similarity_heatmap[i * 2 + 1, j * 2] = neg_similarities_left

                # neg_similarities_right = torch.matmul(neg_emb_i, neg_emb_j.T)  # TODO check if this is correct
                neg_similarities_right = self.myCosineSimilarity(neg_emb_i, neg_emb_j,
                                                                 use_grad=False,
                                                                 to_plot=True)  # TODO check if this is correct
                # if MAX_EMB:
                #     neg_similarities_right, inds = torch.max(neg_similarities_right, dim=0)
                cosine_similarity_heatmap[i * 2 + 1, j * 2 + 1] = neg_similarities_right

        if self.basic_prompts:
            str_prompts = "-single-prompt"
        else:
            str_prompts = "-multiple-prompts"
        heat_map = numpy.array(cosine_similarity_heatmap)
        fig, ax = plt.subplots()
        im, cbar = heatmap(heat_map, [i for i in abbrevviations], [j for j in abbrevviations],
                           ax=ax,
                           cmap="YlGn", cbarlabel="Cosine similarity heatmap" + str_prompts, metric="COS")
        texts = annotate_heatmap(im, valfmt="{x:.2f}")
        fig.tight_layout()
        if epochs > 0 and epoch == 1:
            img_path = 'cosine_similarity_heat_map_fix.png'
            # load the PNG image
            image = read_image(img_path)
            # convert the image to a grid
            self.writer.add_image('visual-embeddings/cosine-similarity Heatmap text-embs', image, 0)

        if epoch == 0 and epochs == 0:
            self.writer.add_figure('visual-embeddings/cosine-similarity Heatmap text-embs', fig, 0)
        if epochs > 0:
            self.writer.add_figure('visual-embeddings/cosine-similarity Heatmap text-embs', fig, epoch)

    @torch.no_grad()
    def myIncremental(self, threshold, iteration):
        if IMAGE_MODEL:
            for (name1, param1), (name2, param2) in zip(self.image_adapter.named_parameters(),
                                                        self.image_adapter_copy.named_parameters()):
                # compare the values of the individual weights
                diff = torch.abs(param1 - param2)
                minimum = diff.min()
                maximum = diff.max()
                # compute the threshold value
                to_reset = minimum + threshold * (maximum - minimum)
                mask = diff < to_reset

                self.n_reset += torch.sum(torch.eq(mask, True))
                self.n_updated += torch.sum(torch.eq(mask, False))
                # reset the updated weights to the old values
                param1.data[mask] = param2.data[mask]
        if TEXT_MODEL:
            for (name1, param1), (name2, param2) in zip(self.text_adapter.named_parameters(),
                                                        self.text_adapter_copy.named_parameters()):
                # compare the values of the individual weights
                diff = torch.abs(param1 - param2)
                minimum = diff.min()
                maximum = diff.max()
                # compute the threshold value
                to_reset = minimum + threshold * (maximum - minimum)
                mask = diff < to_reset

                self.n_reset += torch.sum(torch.eq(mask, True))
                self.n_updated += torch.sum(torch.eq(mask, False))
                # reset the updated weights to the old values
                param1.data[mask] = param2.data[mask]

    @torch.no_grad()
    def profIncremental(self, epoch, epochs, actual_task, threshold):
        if IMAGE_MODEL:
            for (name1, param1), (name2, param2) in zip(self.image_adapter.named_parameters(),
                                                        self.image_adapter_copy.named_parameters()):
                # compare the values of the individual weights
                diff = torch.abs(param1 - param2)
                minimum = diff.min()
                maximum = diff.max()
                # compute the threshold value
                to_reset = minimum + threshold * (maximum - minimum)
                mask = diff < to_reset

                self.n_reset += torch.sum(torch.eq(mask, True))
                self.n_updated += torch.sum(torch.eq(mask, False))
                # reset the updated weights to the old values
                param1.data[mask] = param2.data[mask]
        if TEXT_MODEL:
            for (name1, param1), (name2, param2) in zip(self.text_adapter.named_parameters(),
                                                        self.text_adapter_copy.named_parameters()):
                # compare the values of the individual weights
                diff = torch.abs(param1 - param2)
                minimum = diff.min()
                maximum = diff.max()
                # compute the threshold value
                to_reset = minimum + threshold * (maximum - minimum)
                mask = diff < to_reset

                self.n_reset += torch.sum(torch.eq(mask, True))
                self.n_updated += torch.sum(torch.eq(mask, False))
                # reset the updated weights to the old values
                param1.data[mask] = param2.data[mask]

        print()
        print("number of resets:", self.n_reset.item(), "number of updates:", self.n_updated.item(),
              "percentage resets",
              self.n_reset.item() / (self.n_reset.item() + self.n_updated.item()))
        self.writer.add_scalar("monitor-resets/resets", self.n_reset.item(), (actual_task - 1) * epochs + epoch)
        self.writer.add_scalar("monitor-resets/updates", self.n_updated.item(), (actual_task - 1) * epochs + epoch)
        self.writer.add_scalar("monitor-resets/percentage resets",
                               self.n_reset.item() / (self.n_reset.item() + self.n_updated.item()),
                               (actual_task - 1) * epochs + epoch)
        self.n_reset = 0
        self.n_updated = 0

    @torch.no_grad()
    def model_copy(self):
        if IMAGE_MODEL:
            self.image_adapter_copy = copy.deepcopy(self.image_adapter)
        if TEXT_MODEL:
            self.text_adapter_copy = copy.deepcopy(self.text_adapter)
        self.n_reset = 0
        self.n_updated = 0

    @torch.no_grad()
    def save(self):
        if IMAGE_MODEL:
            torch.save(self.image_adapter, self.writer.log_dir + '/image_adapter.pt')
        if TEXT_MODEL:
            torch.save(self.text_adapter, self.writer.log_dir + '/text_adapter.pt')

    @torch.no_grad()
    def load(self):
        if IMAGE_MODEL:
            self.image_adapter = torch.load(self.writer.log_dir + '/image_adapter.pt')
        if TEXT_MODEL:
            self.text_adapter = torch.save(self.writer.log_dir + '/text_adapter.pt')

    def bert_forward_mean(self, pos_prompt, neg_prompt, use_grad, to_plot=False):

        with torch.set_grad_enabled(use_grad):
            pos_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(pos_prompt, normalize=False)
            pos_prompt_embedding = pos_prompt_embedding.to(self.device)
            if TEXT_MODEL:
                pos_prompt_embedding = self.text_adapter(pos_prompt_embedding)
            assert pos_prompt_embedding.shape[0] == len(pos_prompt)
            if (not self.basic_prompts and not MAX_EMB) or to_plot:
                pos_prompt_embedding = pos_prompt_embedding.mean(dim=0)
            # pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=0, p=2).to(self.device)
            # pos_prompt_embedding = F.normalize(pos_prompt_embedding, dim=-1, p=2).to(self.device)

            neg_prompt_embedding = self.bert_encoder.get_embeddings_from_prompt(neg_prompt, normalize=False)
            neg_prompt_embedding = neg_prompt_embedding.to(self.device)
            if TEXT_MODEL:
                neg_prompt_embedding = self.text_adapter(neg_prompt_embedding)
            assert neg_prompt_embedding.shape[0] == len(neg_prompt)
            if (not self.basic_prompts and not MAX_EMB) or to_plot:
                neg_prompt_embedding = neg_prompt_embedding.mean(dim=0)
            # neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=0, p=2).to(self.device)
            # neg_prompt_embedding = F.normalize(neg_prompt_embedding, dim=-1, p=2).to(self.device)

        return pos_prompt_embedding, neg_prompt_embedding

    def myCosineSimilarity(self, x, y, use_grad, to_plot=False, train=False, pos=None):
        with torch.set_grad_enabled(use_grad):
            # x = F.normalize(x, dim=-1)
            # y = F.normalize(y, dim=0)
            # res = torch.matmul(x, y.T)
            if to_plot:
                res = pairwise_cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))
            elif not MAX_EMB:
                res = pairwise_cosine_similarity(x, y.reshape(1, -1))
            elif MAX_EMB:
                res = pairwise_cosine_similarity(x, y)
                res_mean = torch.mean(res, dim=1)
                res, inds = torch.max(res, dim=1)
                if train:
                    if pos:
                        self.pos_mean_counter += 1
                        self.writer.add_scalar("max-mean-comparison/pos", torch.mean(res - res_mean),
                                               self.pos_mean_counter)
                    else:
                        self.neg_mean_counter += 1
                        self.writer.add_scalar("max-mean-comparison/neg", torch.mean(res - res_mean),
                                               self.neg_mean_counter)
            return res


@torch.no_grad()
def change_values(tensor):
    """
    Takes a 2D torch tensor of float32 with 0 and 1 values and changes 1 to 2 and 0 to -2.
    Returns the modified tensor as a tensor of float32.
    """
    # Check if the input tensor is a 2D tensor
    # todo check se funziona sia per 2d che per 1d
    # if len(tensor.shape) != 2:
    #     raise ValueError("Input tensor must be a 2D tensor.")

    # Create a copy of the input tensor
    new_tensor = tensor.clone()

    # Replace 1 with 2 and 0 with -2
    new_tensor[tensor == 1] = 1  # 2
    new_tensor[tensor == 0] = -1  # -2

    # Convert the tensor to float32
    new_tensor = new_tensor.float()

    return new_tensor.to(tensor.device)
