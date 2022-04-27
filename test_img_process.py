from utils import process_image
import cv2
import numpy as np
import os

out_dir = "outputs/processed_results/"

def test_cleaning():
    src_dir = "outputs/results/"
    file_list = os.listdir(src_dir)
    for i in range(len(file_list)):
        img_fname = file_list[i]
        img = cv2.imread(src_dir + img_fname, 0)
        img = process_image.resize_img_height(img, 256)
        img = process_image.canny_edge_filter(img)
        img = process_image.closing_filter(img)
        cv2.imwrite(out_dir + "test_" + str(i).zfill(2) + ".png", img)

def test_augmentation():
    src_path = "/home/adrian/Projects/hku/stickersearch/outputs/results/StickFaces-Sticker8.png"
    img = cv2.imread(src_path, 0)
    img = process_image.resize_img_height(img, 256)
    img = process_image.canny_edge_filter(img)
    img = process_image.closing_filter(img)
    for i in range(10):
        result_img = process_image.rand_warp(img, padding_bound=3, persp_bound=0.002, rot_bound=8, scale_diff_bound=0.2)
        cv2.imwrite(out_dir + "test_aug_" + str(i).zfill(2) + ".png", result_img)

test_cleaning()