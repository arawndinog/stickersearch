from utils import img_process
import os
import cv2
import numpy as np

def convert_webp_to_png(img_dir, output_dir):
    img_list = os.listdir(img_dir)
    for i in range(len(img_list)):
        img_fname = img_list[i]
        png_path = output_dir + os.path.splitext(img_fname)[0]  + ".png"

        img = cv2.imread(img_dir + img_fname, -1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # img = img_process.resize_img_height(img, 240)
        img_mask = np.expand_dims(img[:,:,3],2)
        img_mask_bool = img_mask.astype(np.bool)
        # img_mask_white = ~img_mask
        img_masked = img[:,:,:3] * img_mask_bool #+ img_mask_white
        cv2.imwrite(png_path, img_masked)
    return

if __name__ == "__main__":
    convert_webp_to_png("outputs/stick-faces/", "outputs/results/")