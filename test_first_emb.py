import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    precision_recall_curve
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm

from DataRetrieval import DataRetrieval
from Trainer import Trainer
from health_multimodal.image import get_biovil_resnet
from health_multimodal.text.utils import get_cxr_bert_inference
from models import myLinearModel

#  SET REPRODUCIBILITY
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

# train
# img
# labels OK

# val
# img
# labels OK

# test
# img
# labels OK

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on:", device)

    resnet_path = "C:\\Users\\mistr\\OneDrive\\Desktop\\mcs_only_local\\hi-ml\\hi-ml-multimodal\\src\\biovil_image_resnet50_proj_size_128.pt"
    resnet50 = get_biovil_resnet(pretrained=resnet_path)
    resnet50.train(mode=False, my_freeze=True)
    resnet50.eval()
    if not resnet50.training:
        print("Res-Net is in eval mode")
    resnet50.to(device)
    stop = True
    train_dataset = torch.load("embeddingDataset\\train\\512-chex-not-normalize\\embeddings_dataset_final_old.pt")
    # train_dataset = torch.load("embeddingDataset\\test\\512-chex-not-normalize\\embeddings_dataset_final_old.pt")
    # train_dataset = torch.load("embeddingDataset\\val\\512-chex-not-normalize\\embeddings_dataset_final_old.pt")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=None, batch_size=1,
                                               shuffle=False,
                                               num_workers=1, pin_memory=False, drop_last=False)

    labels_list_1 = torch.empty(0, 5)
    images_list_1 = torch.empty(0, 128)

    for emb_1, label_1 in tqdm(train_loader):
        labels_list_1 = torch.cat([labels_list_1, label_1], dim=0)
        images_list_1 = torch.cat([images_list_1, emb_1], dim=0)

    img_size = 512
    img_dir = "D:\\CheXpert-v1.0\\CheXpert-v1.0\\"  # xxx train
    # img_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\"  # xxx test or val

    labels_dir = "D:\\CheXpert-v1.0\\CheXpert-v1.0\\train_visualCheXbert_fixed.csv"  # xxx train
    # labels_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\test_labels.csv"  # xxx test
    # labels_dir = "C:\\Users\\mistr\\OneDrive\\Desktop\\real-chexpert\\val_labels.csv"  # xxx test

    chexpert_loader = DataRetrieval(dataset="chexpert", labels_dir=labels_dir,
                                    img_dir=img_dir, batch_size=1, perc_dataset=1,
                                    size=img_size, s=0, verbose=True, num_workers=1).loader

    labels_list_2 = torch.empty(0, 5)
    images_list_2 = torch.empty(0, 128)
    # for img_2, label_2 in tqdm(chexpert_loader):
    #     img_2 = img_2.to(device)
    #     emb_2 = resnet50(img_2)
    #     break

    for img_2, label_2 in tqdm(chexpert_loader):
        labels_list_2 = torch.cat([labels_list_2, label_2], dim=0)
        img_2 = img_2.to(device)
        images_list_2 = torch.cat([images_list_2, resnet50(img_2).detach().to('cpu')], dim=0)


    # print(torch.max(torch.max(label_1-label_2, dim=0), dim=0))
    debug = 0
