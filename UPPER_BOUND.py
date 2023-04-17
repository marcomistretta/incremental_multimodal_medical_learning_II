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
from models import Adapter

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
    lr = 0.0001  # 0.1  # 0.0001, 1, 30
    epochs = 10  # 10  # todo test with epochs 1 (online scenario)
    basic_prompts = False  # False-->multiple True-->single
    chex_competition = True  # True, False
    xrays_position = "all"  # "all", "frontal", "lateral"
    loss_name = "standard"  # standard, opzione2, opzione2variant
    writer, class_names, train_loader, val_loader, test_loader, prompts = Trainer.preprocessing(chex_competition,
                                                                                                xrays_position,
                                                                                                basic_prompts,
                                                                                                batch_size, lr, epochs,
                                                                                                loss_name=loss_name)
    bert_encoder = get_cxr_bert_inference()
    image_adapter = Adapter().to(device)
    text_adapter = Adapter().to(device)
    params = list(image_adapter.parameters()) + list(text_adapter.parameters())
    optimizer = optim.Adam(params, lr=lr)  # todo tune optimizer

    criterion = nn.BCEWithLogitsLoss()  # nn.BCEWithLogitsLoss() nn.CrossEntropyLoss

    trainer = Trainer(image_adapter, bert_encoder, text_adapter, prompts, class_names, device, writer, loss_name)

    pre_test = False
    if pre_test:
        trainer.pre_test(test_loader, basic_prompts)
        print(1 / 0)

    # XXX run
    CONTINUAL_LEARNING = None  # "myCL", "profCL"
    threshold = 1
    # lr_scheduler = ExponentialLR(optimizer, gamma=(0.001 / 0.1) ** (1 / 370))
    if CONTINUAL_LEARNING is not None:
        print("**** CONTINUAL LEARNING ****")
        print("--->", CONTINUAL_LEARNING)
    else:
        print("**** UPPER BOUND ****")

    for epoch in range(1, epochs + 1):
        trainer.train(train_loader, optimizer, criterion, epoch, basic_prompts, CONTINUAL_LEARNING, threshold)
        trainer.val(val_loader, optimizer, criterion, epoch, basic_prompts, epochs)
        trainer.test(test_loader, optimizer, criterion, epoch, basic_prompts, epochs)
