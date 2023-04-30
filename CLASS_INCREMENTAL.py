'''
Questo può essere diviso in quanti task mi pare...
Il numero di task == numero di "epoche"
Split senza intersezione e lasciando val e test invariati
Qua posso provare tutte le loss che voglio

'''
import copy
import playsound
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_curve
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from Trainer import Trainer
from health_multimodal.text.utils import get_cxr_bert_inference
from models import myLinearModel

# xxx SET REPRODUCIBILITY
seed_value = 27
torch.manual_seed(seed_value)
import random

random.seed(seed_value)
np.random.seed(seed_value)
# xxx

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on:", device)

    batch_size = 6144  # 4096, 6144 8192, 8192 (val e test sono settati di default a 1024 per avere dei plot meno rumorosi)
    lr = 0.1  # 0.001  # 0.1  # 0.0001, 1, 30
    epochs = 10  # 10
    single_prompt = False  # False-->multiple True-->single
    chex_competition = True  # True, False
    xrays_position = "all"  # "all", "frontal", "lateral"
    loss_name = "standard"  # "standard", "opzione2" "opzione2variant", "bce"

    n_tasks = 5
    tasks_order = [0, 1, 2, 3, 4]

    CONTINUAL_LEARNING = None  # "myCL"  # "profCL"  # "myCL"  # "myCL", "profCL", None
    threshold = 0.01
    ratio = True
    adder = 0.001
    threshold_scheduling = True
    mode = "class-pos-neg"  # "class-pos-neg" / "class-pos"
    writer, class_names, train_loader, val_loader, test_loader, prompts = Trainer.preprocessing_class_incremental(
        chex_competition, xrays_position, single_prompt, batch_size, lr,
        epochs, loss_name, mode, CONTINUAL_LEARNING, ratio,
        threshold, threshold_scheduling, adder)

    criterion = nn.BCEWithLogitsLoss()
    trainer = Trainer(single_prompt, prompts, class_names, loss_name, lr, device, writer)

    last_batch = 0
    count = 0
    try:
        for actual_task in range(1, n_tasks + 1):
            for epoch in range(1, epochs + 1):
                count += 1
                threshold = threshold + adder
                if threshold_scheduling and CONTINUAL_LEARNING is not None:
                    writer.add_scalar("monitor-resets/threshold-scheduling", threshold, count)
                if CONTINUAL_LEARNING == "profCL":
                    trainer.model_copy()
                last_batch = trainer.train_class_incremental(train_loader[actual_task - 1], criterion, epoch,
                                                             CONTINUAL_LEARNING, threshold,
                                                             tasks_order[actual_task - 1],
                                                             last_batch)
                if CONTINUAL_LEARNING == "profCL":
                    trainer.profIncremental(epoch, epochs, actual_task, threshold)
            trainer.val(val_loader, criterion, actual_task, epochs, mode=mode, tasks_order=tasks_order)
            trainer.test(test_loader, criterion, actual_task, epochs, mode=mode, tasks_order=tasks_order)
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        # Play a sound to notify the end of the execution
        playsound.playsound("mixkit-correct-answer-tone-2870.wav")
