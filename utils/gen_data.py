import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os
import numpy as np
import random
from utils import process_image
import albumentations as A

class StickerDataset(Dataset):
    def __init__(self, img_dir_list, input_size, augmentation=False):
        self.input_size = input_size
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

    def __len__(self):
        return len(self.img_labels)

    def max_label(self):
        return max(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = self.img_list[idx]

        img = cv2.resize(img, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        img = img.float()
        img = img/255.0
        
        label = idx
        return img, label, img_path


class StickerDatasetTriplet(Dataset):
    def __init__(self, img_dir_list, input_size, augment=False):
        ''' dataset folder structure:
        batch_X/
            cleaned/        anchor images
            canny/          mimic images
            hed/
            photosketch/
            ...
        '''
        self.input_size = input_size
        self.augment = augment
        self.img_anc_pool = []
        self.img_pos_pool = []      # list of list
        self.img_anc_pool_paths = []
        for img_dir in img_dir_list:
            img_anc_dir = "cleaned/"
            img_pos_dir_list = sorted(os.listdir(img_dir))
            img_pos_dir_list.remove(img_anc_dir.split("/")[0])
            img_anc_list = sorted(os.listdir(img_dir + img_anc_dir))
            for i in range(len(img_anc_list)):
                img_fname = img_anc_list[i]
                if os.path.splitext(img_fname)[1] == ".png":
                    img_anc_fullpath = img_dir + img_anc_dir + img_fname
                    img_anc = cv2.imread(img_anc_fullpath, 0)
                    self.img_anc_pool.append(img_anc)
                    self.img_anc_pool_paths.append(img_anc_fullpath)
                    img_pos_list = []
                    for img_pos_dir in img_pos_dir_list:
                        img_pos_fullpath = img_dir + img_pos_dir + "/" + img_fname
                        img_pos = cv2.imread(img_pos_fullpath, 0)
                        img_pos_list.append(img_pos)
                    self.img_pos_pool.append(img_pos_list)

        self.img_labels = list(range(len(self.img_anc_pool)))
        self.elastic_transform = A.ElasticTransform(p=0.5, alpha=1, sigma=50, alpha_affine=50, border_mode=cv2.BORDER_CONSTANT)

    def __len__(self):
        return len(self.img_labels)

    def max_label(self):
        return max(self.img_labels)

    def __getitem__(self, idx):
        img_anc = self.img_anc_pool[idx]
        img_anc_path = self.img_anc_pool_paths[idx]
        img_pos_list = self.img_pos_pool[idx]
        # img_pos = random.choice(img_pos_list)
        img_pos = img_pos_list[0]

        img_other_indices = self.img_labels.copy()
        img_other_indices.remove(idx)
        img_neg_idx = np.random.choice(img_other_indices)
        img_neg_list = [self.img_anc_pool[img_neg_idx]] + self.img_pos_pool[img_neg_idx]
        img_neg = random.choice(img_neg_list)

        if self.augment:
            if np.random.rand() < 0.5:
                # img_pos = process_image.rand_warp(img_pos, padding_bound=3, persp_bound=0.001, rot_bound=2, scale_diff_bound=0.2)
                img_pos = process_image.rand_rotate(img_pos, 2)
                img_pos = process_image.rand_translate_by_factor(img_pos, 0.1)
                img_pos = self.elastic_transform(image=img_pos)["image"]

            if np.random.rand() < 0.5:
                # img_neg = process_image.rand_warp(img_neg, padding_bound=3, persp_bound=0.001, rot_bound=2, scale_diff_bound=0.2)
                img_neg = process_image.rand_rotate(img_neg, 2)
                img_neg = process_image.rand_translate_by_factor(img_neg, 0.1)
                img_neg = self.elastic_transform(image=img_neg)["image"]

        img_anc = cv2.resize(img_anc, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img_pos = cv2.resize(img_pos, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        img_neg = cv2.resize(img_neg, dsize=(self.input_size, self.input_size), interpolation=cv2.INTER_AREA)

        img_anc = np.expand_dims(img_anc, 0)
        img_pos = np.expand_dims(img_pos, 0)
        img_neg = np.expand_dims(img_neg, 0)

        img_anc = torch.from_numpy(img_anc)
        img_pos = torch.from_numpy(img_pos)
        img_neg = torch.from_numpy(img_neg)

        img_anc = img_anc.float()
        img_pos = img_pos.float()
        img_neg = img_neg.float()

        img_anc = img_anc/255.0
        img_pos = img_pos/255.0
        img_neg = img_neg/255.0
        
        label = idx
        neg_label = img_neg_idx
        return img_anc, img_pos, img_neg, label, neg_label, img_anc_path