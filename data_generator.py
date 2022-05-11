from socket import IOCTL_VM_SOCKETS_GET_LOCAL_CID
import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
np.set_printoptions(threshold=np.inf)

class StickerDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.img_labels = list(range(len(self.img_list)))

    def __len__(self):
        return len(self.img_labels)

    def max_label(self):
        return max(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        image = cv2.imread(img_path, 0)
        # image = cv2.resize(image, (256, 256))
        image = cv2.resize(image, (64, 64))
        image = ~image / 255.0   # make pixels to be max 1
        # cv2.imwrite("outputs/test.png", image*255)
        image = np.expand_dims(image, 0)
        image = image.astype(np.float32)
        label = idx
        return image, label