import cv2
import numpy as np

def resize_img_height(img, new_h):
    img_h = img.shape[0]
    img_h_scale = new_h/img_h
    new_w = int(img.shape[1] * img_h_scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    return img_resized

def canny_edge_filter(img):
    img = cv2.Canny(img,100,200)
    return img

def closing_filter(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img, kernel, iterations = 2)
    img = cv2.erode(img, kernel, iterations = 2)
    return img