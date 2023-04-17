'''
Questo può essere diviso in quanti task mi pare...
Il numero di task == numero di "epoche"
Split senza intersezione e lasciando val e test invariati
Qua posso provare tutte le loss che voglio

'''
import copy

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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on:", device)

    batch_size = 6144  # 4096, 6144 8192, 8192 (val e test sono settati di default a 1024 per avere dei plot meno rumorosi)
    lr = 0.0001  # 0.001  # 0.1  # 0.0001, 1, 30
    n_tasks = 5  # xxx == tasks
    tasks_order = [0, 1, 2, 3, 4]
    # tasks_order = [2, 0, 3, 1, 4]
    epochs = 10
    # N = 3
    # arr_repeated = np.repeat(arr, N)

    # XXX run
    CONTINUAL_LEARNING = None  # "myCL"  # "profCL"  # "profCL"  # None "myCL", "profCL"
    threshold = 0.5  # 6e-3
    ratio = True
    loss_name = "standard"  # "standard", "opzione2" "opzione2variant" # todo implement bce
    basic_prompts = False  # False-->multiple True-->single
    chex_competition = True  # True, False
    xrays_position = "all"  # "all", "frontal", "lateral"

    writer, class_names, train_loader, val_loader, test_loader, prompts = Trainer.preprocessing_class_incremental_two_class(
        loss_name,
        chex_competition,
        xrays_position,
        basic_prompts,
        batch_size, lr, epochs, CONTINUAL_LEARNING, threshold, ratio, tasks_order)

    # xxx notare che non c'è bisogno di una resnet
    model = myLinearModel().to(device)
    cxr_bert = get_cxr_bert_inference()

    criterion = nn.BCEWithLogitsLoss()
    trainer = Trainer(model, cxr_bert, prompts, class_names, device, writer, change_labels=False, loss_name=loss_name)

    optimizer = optim.Adam(model.parameters(), lr=lr)  # todo tune optimizer
    last_batch = 0

    f1_heat_map = torch.zeros((n_tasks, n_tasks))
    auroc_heat_map = torch.zeros((n_tasks, n_tasks))  # todo

    for actual_task in range(1, n_tasks + 1):
        for epoch in range(1, epochs + 1):
            if CONTINUAL_LEARNING == "profCL":
                # make a copy of the original image_adapter before starting the training loop
                model_copy = copy.deepcopy(model)
                # define the threshold for parameter updates

            last_batch = trainer.train_class_incremental(train_loader[actual_task - 1], optimizer, criterion, epoch,
                                                         basic_prompts,
                                                         tasks_order[actual_task - 1], CONTINUAL_LEARNING,
                                                         threshold, last_batch)
            if CONTINUAL_LEARNING == "profCL":
                # at the end of the epoch, compare the updated image_adapter with the image_adapter copy
                n_reset = 0
                n_updated = 0
                for (name1, param1), (name2, param2) in zip(model.named_parameters(), model_copy.named_parameters()):
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
                writer.add_scalar("monitor-resets/resets", n_reset.item(), (actual_task - 1) * epochs + epoch)
                writer.add_scalar("monitor-resets/updates", n_updated.item(), (actual_task - 1) * epochs + epoch)
                writer.add_scalar("monitor-resets/percentage resets",
                                  n_reset.item() / (n_reset.item() + n_updated.item()),
                                  (actual_task - 1) * epochs + epoch)
        trainer.val(val_loader, optimizer, criterion, tasks_order[actual_task - 1], basic_prompts, tasks_order[-1])
        f1_heat_map = trainer.test(test_loader, optimizer, criterion, tasks_order[actual_task - 1], basic_prompts,
                                   tasks_order[-1], mode="class",
                                   tasks_order=tasks_order, f1_heat_map=f1_heat_map)
