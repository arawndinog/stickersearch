import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
import numpy as np

class StickerDataset(Dataset):
    def __init__(self, img_dir_list, input_size, augmentation=False):
        self.img_path_list = []
        for img_dir in img_dir_list:
            img_relative_path_list = sorted(os.listdir(img_dir))
            self.img_path_list += [img_dir + img_relative_path for img_relative_path in img_relative_path_list]
        self.img_list = []
        for img_path in self.img_path_list:
            img = cv2.imread(img_path, 0)
            self.img_list.append(img)

        self.img_labels = list(range(len(self.img_path_list)))
        self.augmentation = augmentation

        self.augmentations = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.8, 1), shear=None),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.Resize((input_size, input_size))
        ])

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size))
        ])

    def __len__(self):
        return len(self.img_labels)

    def max_label(self):
        return max(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = self.img_list[idx]
        if self.augmentation:
            img = self.augmentations(img)
        else:
            img = self.transforms(img)
        
        label = idx
        return img, label, img_path


class StickerDatasetTriplet(Dataset):
    def __init__(self, img_dir_list, img_mimic_dir_list, input_size):
        self.img_path_list = []
        for img_dir in img_dir_list:
            img_relative_path_list = sorted(os.listdir(img_dir))
            self.img_path_list += [img_dir + img_relative_path for img_relative_path in img_relative_path_list]
        self.img_list = []
        for img_path in self.img_path_list:
            img = cv2.imread(img_path, 0)
            self.img_list.append(img)

        self.img_labels = list(range(len(self.img_path_list)))

        self.augmentations = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.8, 1), shear=None),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.Resize((input_size, input_size))
        ])

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((input_size, input_size))
        ])

    def __len__(self):
        return len(self.img_labels)

    def max_label(self):
        return max(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = self.img_list[idx]

        img_anc = self.transforms(img)
        img_pos = self.augmentations(img)

        img_other_indices = self.img_labels.copy()
        img_other_indices.remove(idx)
        img_neg_idx = np.random.choice(img_other_indices)
        img_neg = self.img_list[img_neg_idx]
        if np.random.rand() < 0.5:
            img_neg = self.augmentations(img_neg)
        else:
            img_neg = self.transforms(img_neg)
        
        label = idx
        neg_label = img_neg_idx
        return img_anc, img_pos, img_neg, label, neg_label, img_path