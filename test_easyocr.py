import easyocr
import os
import cv2

reader = easyocr.Reader(['ch_tra','en']) # this needs to run only once to load the model into memory

img_dir = "dataset/stickers_png/batch_2/cleaned/"
output_path = "outputs/easyocr_test.txt"
output_file = open(output_path, 'w')
img_list = sorted(os.listdir(img_dir))
for img_fname in img_list:
    img_path = img_dir + img_fname
    # img = cv2.imread(img_path, 1)
    # img = 255-img
    result = reader.readtext(img_path, detail = 0)
    result = " ".join(result)
    print(img_fname, result)
    output_file.write(result + "\n")