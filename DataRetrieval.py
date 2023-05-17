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
from new_texts_prompts import generate_chexpert_class_prompts

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

    print("color distortion intensity:", s)


class DataRetrieval:

    def __init__(self, dataset, labels_dir, img_dir, batch_size, perc_dataset=1, size=512, s=0, verbose=False,
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
    # def get_pretext_pipeline(self, size, s=0.5):
    #     augmentation = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Grayscale(num_output_channels=3),
    #         transforms.Resize(size=size),
    #         transforms.CenterCrop(size),
    #         transforms.ToTensor()]
    #         # self.normalize
    #     )
    #
    #     return augmentation


def get_bio_vil_pipeline(size):
    augmentation = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
         ExpandChannels()])

    return augmentation


def get_prompts(single, class_list, opz):
    if single:
        print("*** SINGLE PROMPTS ***")
        return _single_prompts(class_list, opz)
    else:
        print("*** MULTIPLE PROMPTS ***")
        return _multiple_prompts(class_list, opz)


def _single_prompts(class_list, opz):
    print("*** Basic Prompting ***")
    my_prompts = {}
    if opz == 0:
        # xxx SINGLE
        print()
        print("--------------- SINGLE UN POS e UN NEG per malattia ---------------")
        for c in class_list:
            my_prompts[c] = {
                "positive": [f"Findings suggesting {c}"],
                "negative": [f"No evidence of {c}"],
            }
    elif opz == 1:
        # xxx UGUALE PER TUTTI
        print()
        print("--------------- SINGLE, c'è qualcosa / non c'è qualcosa ---------------")
        for c in class_list:
            my_prompts[c] = {
                "positive": [f"Findings suggesting disease"],
                "negative": [f"No evidence of disease"],
            }
    elif opz == 2:
        # xxx ANIMALI
        print()
        print("--------------- SINGLE ANIMALE ---------------")
        my_prompts[class_list[0]] = {
            "positive": ["Findings suggesting dog"],
            "negative": ["No evidence of dog"],
        }
        my_prompts[class_list[1]] = {
            "positive": ["Findings suggesting cat"],
            "negative": ["No evidence of cat"],
        }
        my_prompts[class_list[2]] = {
            "positive": ["Findings suggesting cow"],
            "negative": ["No evidence of cow"],
        }
        my_prompts[class_list[3]] = {
            "positive": ["Findings suggesting sheep"],
            "negative": ["No evidence of sheep"],
        }
        my_prompts[class_list[4]] = {
            "positive": ["Findings suggesting turtle"],
            "negative": ["No evidence of turtle"],
        }
    elif opz == 3:
        # xxx SINGLE CHEX MEDCLIP PROMPTS
        print()
        print("--------------- SINGLE CHEX MEDCLIP PROMPTS ---------------")
        my_prompts = generate_chexpert_class_prompts(train_logit_diff=False, n=1)
    else:
        raise Exception("opz not found")
    return my_prompts


def _multiple_prompts(class_list, opz):
    print("*** Multiple Prompting ***")
    my_prompts = {}
    if opz == 0:
        # xxx MULTIPLE
        print()
        print("--------------- MULTIPLE VANILLA ---------------")
        for c in class_list:
            my_prompts[c] = {
                "positive": [f"Findings consistent with {c}", f"Findings suggesting {c}",
                             f"This opacity can represent {c}", f"Findings are most compatible with {c}"],
                "negative": [f"There is no {c}", f"No evidence of {c}",
                             f"No evidence of acute {c}", f"No signs of {c}"]
            }
    # todo opz 1
    elif opz == 2:
        # xxx MULTIPLE ANIMALE
        print()
        print("--------------- 4 per ANIMALE ---------------")
        animals = ["dog", "cat", "cow", "sheep", "turtle"]
        for idx, c in enumerate(class_list):
            my_prompts[c] = {
                "positive": [f"Findings consistent with {animals[idx]}", f"Findings suggesting {animals[idx]}",
                             f"This opacity can represent {animals[idx]}", f"Findings are most compatible with {animals[idx]}"],
                "negative": [f"There is no {animals[idx]}", f"No evidence of {animals[idx]}",
                             f"No evidence of acute {animals[idx]}", f"No signs of {animals[idx]}"]
            }
    elif opz == 3:
        # xxx MULIPLE CHEX MEDCLIP PROMPTS
        print()
        print("--------------- CHEX MEDCLIP PROMPTS ---------------")
        my_prompts = generate_chexpert_class_prompts(train_logit_diff=True, n=4)
    else:
        raise Exception("opz:", opz, "not found")
    return my_prompts