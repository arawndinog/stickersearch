from utils import process_image
import os
import cv2
import numpy as np

def convert_webp_to_png(img_dir: str, output_dir: str) -> None:
    img_list = os.listdir(img_dir)
    for i in range(len(img_list)):
        img_fname = img_list[i]
        png_path = output_dir + "img_" + str(i).zfill(3)  + ".png"
        img = cv2.imread(img_dir + img_fname, -1)
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
        cv2.imwrite(png_path, img)
    return

if __name__ == "__main__":
    # convert_webp_to_png("outputs/stick-faces/", "outputs/results/")
    # convert_webp_to_png("dataset/stickers_webp/batch_2/", "dataset/stickers_png/batch_2/")
    convert_webp_to_png("dataset/stickers_webp/batch_2/", "dataset/stickers_png/batch_2_cleaned/")