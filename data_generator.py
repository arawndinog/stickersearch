import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import os

class StickerDataset(Dataset):
    def __init__(self, img_dir, input_size, augmentation=False):
        self.img_dir = img_dir
        self.img_list = sorted(os.listdir(img_dir))
        self.img_labels = list(range(len(self.img_list)))
        self.augmentation = augmentation

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
        img_path = os.path.join(self.img_dir, self.img_list[idx])

        img = cv2.imread(img_path, 0)
        if self.augmentation:
            img = self.augmentations(img)
        else:
            img = self.transforms(img)
        
        label = idx
        return img, label, img_path