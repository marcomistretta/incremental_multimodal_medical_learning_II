import os
import random
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.io import read_image

# from csv import reader

plt.ion()


# class TwoCropsTransform:
#
#     def __init__(self, base_transform):
#         self.base_transform = base_transform
#
#     def __call__(self, x):
#         q = self.base_transform(x)
#         k = self.base_transform(x)
#         return [q, k]
class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)


def plotImage(image):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    image.cpu().numpy().transpose((1, 2, 0))
    image = normalize.std * image + normalize.mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.show()


# Load the CSV file
# def load_csv(filename):
#     data = list()
#     # Open file in read mode
#     file = open(filename, "r")
#     # Reading file
#     lines = reader(file)
#     csv_reader = reader(file)
#     for row in csv_reader:
#         if not row:
#             continue
#         data.append(row)
#
#     return data


class CustomDataset(Dataset):
    def __init__(self, dataset, annotations_file, img_dir, transform=None):
        # self.img_labels = load_csv(annotations_file)
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.dataset = dataset

        self.file_extension = ""
        # self.label_names = ['Pleural Effusion', 'Pneumothorax', 'Atelectasis', 'Pneumonia', 'Consolidation']
        self.label_names = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        self.n_classes = 5

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir+ self.img_labels[idx][0])
        img_path = str(self.img_dir) + str(self.img_labels.iloc[idx, 0])
        # print(img_path)
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        # plotImage(image) todo fix
        if self.dataset == "chexpert":
            label = torch.tensor(self.img_labels.loc[idx, self.label_names].values.astype(np.float32))
            # print(label)

        return image, label


#
# class GaussianBlur(object):
#
#     def __init__(self, sigma=None):
#         if sigma is None:
#             sigma = [.1, 2.]
#         self.sigma = sigma
#
#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x


def _debug_info(dataset, labels_dir, img_dir, batch_size, perc_dataset, size, s):
    print()
    print('-' * 30)
    print("Allocating DataRetrieval")
    print("Dataset:", dataset)
    print("percentage dataset:", perc_dataset)
    print("labels dir:", labels_dir)
    print("img dir:", img_dir)
    print("batch size:", batch_size)
    print("size of reformat:", size)

    if s != 0:
        print("color distortion intensity:", s)


class DataRetrieval:

    def __init__(self, dataset, labels_dir, img_dir, batch_size, perc_dataset=1, size=224, s=0.5, verbose=False,
                 num_workers=4):
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                       std=[0.229, 0.224, 0.225])
        if dataset != "chexpert":
            raise Exception("dataset not found")
        self.dataset = dataset

        if verbose:
            _debug_info(dataset, labels_dir, img_dir, batch_size, perc_dataset, size, s)

        transform = get_bio_vil_pipeline(size)

        self.dataset = CustomDataset(dataset=dataset,
                                     annotations_file=labels_dir,
                                     img_dir=img_dir,
                                     transform=transform
                                     )

        # self.sampler = torch.utils.data.Subset(self.dataset,
        #                                        range(int(len(self.dataset) * perc_dataset)))

        # xxx shuffle false
        self.loader = torch.utils.data.DataLoader(dataset=self.dataset, sampler=None, batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers, pin_memory=True, drop_last=False)

    # def get_color_distortion(self, s=1.0):
    #     color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    #     rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    #     rnd_gray = transforms.RandomGrayscale(p=0.2)
    #     color_distort = transforms.Compose([
    #         rnd_color_jitter, rnd_gray])
    #     return color_distort
    # todo uncomment
    # def get_pretext_pipeline(self, size, s=0.5):
    #     augmentation = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Grayscale(num_output_channels=3), # todo uncooment
    #         transforms.Resize(size=size),
    #         transforms.CenterCrop(size),
    #         transforms.ToTensor()]
    #         # self.normalize
    #     )
    #
    #     return augmentation


def get_bio_vil_pipeline(size):
    augmentation = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(), ExpandChannels()])

    return augmentation


def basic_create_prompts(class_list):
    prompts = {}
    for c in class_list:
        prompts[c] = {
            "positive": [f"Findings suggesting {c}"],
            "negative": [f"No evidence of {c}"],
        }
    return prompts


def create_prompts(class_list):
    prompts = {}
    for c in class_list:
        prompts[c] = {
            "positive": [f"Findings consistent with {c}", f"Findings suggesting {c}",
                         f"This opacity can represent {c}", f"Findings are most compatible with {c}"],
            "negative": [f"There is no {c}", f"No evidence of {c}",
                         f"No evidence of acute {c}", f"No signs of {c}"]
        }
    return prompts

if __name__ == '__main__':
    class_list = ["Pleural Effusion", "Pneumothorax", "Atelectasis", "Pneumonia", "Consolidation"]
    prompts = create_prompts(class_list)
    print(prompts)