from utils import img_process
import cv2
import os

if __name__ == "__main__":
    src_dir = "outputs/results/"
    out_dir = "outputs/processed_results/"
    file_list = os.listdir(src_dir)
    for i in range(len(file_list)):
        img_fname = file_list[i]
        img = cv2.imread(src_dir + img_fname, 0)
        img = img_process.resize_img_height(img, 256)
        img = img_process.canny_edge_filter(img)
        img = img_process.closing_filter(img)
        cv2.imwrite(out_dir + "test" + "_" + str(i).zfill(2) + ".png", img)