from utils import process_image
import cv2
import numpy as np
import os
from torchvision import transforms

def test_cleaning():
    src_dir = "dataset/stickers_webp/batch_1/"
    out_dir = "outputs/processed_results2/"
    file_list = os.listdir(src_dir)
    for i in range(len(file_list)):
        img_fname = file_list[i]
        img = cv2.imread(src_dir + img_fname, -1)
        # extract png mask
        img_mask = img[:,:,3]
        # refine mask with blue etc.
        img_mask = cv2.erode(img_mask, (5,5), iterations = 2)
        img_mask = cv2.GaussianBlur(img_mask, (3,3), 1)
        img_mask = np.expand_dims(img_mask, 2)
        # change mask to float to allow gray
        img_mask_intensity = img_mask.astype(float)/255.0
        # add bg to contrast text etc.
        img_bg = 255 - img_mask
        # result color image
        img = img[:,:,:3] * img_mask_intensity + img_bg
        # convert image to grayscale
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # flip colors to make bg black for better augmentation
        img = 255-img
        cv2.imwrite(out_dir + "test_" + str(i).zfill(2) + ".png", img)

def test_pt_augmentation():
    src_dir = "dataset/stickers_png/batch_1_cleaned/"
    out_dir = "outputs/processed_results3/"
    file_list = os.listdir(src_dir)

    pt_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.8, 1), shear=None),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.Resize((128,128))
    ])

    for i in range(len(file_list)):
        img_fname = file_list[i]
        img = cv2.imread(src_dir + img_fname, 0)
        img = pt_transforms(img)
        img = img.numpy()
        img = img.squeeze()
        img = img * 255.0
        cv2.imwrite(out_dir + "test_" + str(i).zfill(2) + ".png", img)

def test_filters():
    src_dir = "dataset/stickers_webp/batch_1/"
    out_dir = "outputs/processed_results2/"
    file_list = os.listdir(src_dir)
    for i in range(len(file_list)):
        img_fname = file_list[i]
        img = cv2.imread(src_dir + img_fname, 0)
        # img = process_image.resize_img_height(img, 256)
        # img = process_image.canny_edge_filter(img)
        # img = process_image.closing_filter(img)
        cv2.imwrite(out_dir + "test_" + str(i).zfill(2) + ".png", img)

def test_cv_augmentation():
    src_path = "/home/adrian/Projects/hku/stickersearch/outputs/results/StickFaces-Sticker8.png"
    out_dir = "outputs/processed_results2/"
    img = cv2.imread(src_path, 0)
    img = process_image.resize_img_height(img, 256)
    img = process_image.canny_edge_filter(img)
    img = process_image.closing_filter(img)
    for i in range(10):
        result_img = process_image.rand_warp(img, padding_bound=3, persp_bound=0.002, rot_bound=8, scale_diff_bound=0.2)
        cv2.imwrite(out_dir + "test_aug_" + str(i).zfill(2) + ".png", result_img)

test_cleaning()