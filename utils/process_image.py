import cv2
import numpy as np
import os

def get_img_list(img_dir_path: str, color_mode: int):
    src_dir = img_dir_path
    file_list = sorted(os.listdir(src_dir))
    img_list = []
    for i in range(len(file_list)):
        img_fname = file_list[i]
        img = cv2.imread(src_dir + img_fname, color_mode)
        img_list.append(img)
    return img_list

def resize_img_height(img: np.ndarray, new_h: int) -> np.ndarray:
    img_h = img.shape[0]
    img_h_scale = new_h/img_h
    new_w = int(img.shape[1] * img_h_scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    return img_resized

def canny_edge_filter(img: np.ndarray) -> np.ndarray:
    img = cv2.Canny(img,100,200)
    return img

def closing_filter(img: np.ndarray) -> np.ndarray:
    kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img, kernel, iterations = 2)
    img = cv2.erode(img, kernel, iterations = 2)
    return img

def fill_holes(img: np.ndarray) -> np.ndarray:
    return img

def extend_canvas(img: np.ndarray, output_size) -> np.ndarray:
    img_shape = img.shape
    # if either output width is smaller than orig width, do nothing
    if (output_size[0] < img_shape[0]) or (output_size[1] < img_shape[1]):
        return img
    # if input and target size are same, do nothing
    if (output_size[0] == img_shape[0]) and (output_size[1] == img_shape[1]):
        return img
    # Init empty img of final size with same number of channels as orig img
    output_shape = list(img_shape[:])
    output_shape[0] = output_size[0]
    output_shape[1] = output_size[1]
    final_img = np.zeros(output_shape, img.dtype)
    # Calc border offsets
    row_offset = (output_shape[0] - img_shape[0])//2
    col_offset = (output_shape[1] - img_shape[1])//2
    # Paste orig img
    final_img[row_offset:row_offset+img_shape[0], col_offset:col_offset+img_shape[1]] = img
    return final_img

def extend_canvas_square(img: np.ndarray) -> np.ndarray:
    return extend_canvas(img, (max(img.shape[0], img.shape[1]), max(img.shape[0], img.shape[1])))

def crop_border(img: np.ndarray, keep_rows: bool=False, keep_cols: bool=False) -> np.ndarray:
    # Ensure it is white on black background
    # Assume img is already binarized
    if not keep_rows:
        # Calculate histogram on row axis
        row_thresh = np.sum(img, axis=1)
        # Merge histogram on channel axis
        if len(row_thresh.shape) == 2:
            row_thresh = np.sum(row_thresh, axis=1)
        # Find boundaries
        row_hist = np.argwhere(row_thresh>0)
        # Ensure histogram is >1 on both sides, otherwise return unmodified
        if row_hist.size > 1:
            min_row = np.amin(row_hist)
            max_row = np.amax(row_hist)
            img = img[min_row:max_row+1, :]

    if not keep_cols:
        # repeat the same steps for cols
        col_thresh = np.sum(img, axis=0)
        if len(col_thresh.shape) == 2:
            col_thresh = np.sum(col_thresh, axis=1)
        col_hist = np.argwhere(col_thresh>0)
        if col_hist.size > 1:
            min_col = np.amin(col_hist)
            max_col = np.amax(col_hist)
            img = img[:, min_col:max_col+1]

    return img

def translate_by_pixel(img: np.ndarray, row_translate_pixel: int=1, col_translate_pixel: int=1) -> np.ndarray:
    rows = img.shape[0]
    cols = img.shape[1]
    M = np.float32([[1, 0, row_translate_pixel],[0, 1, col_translate_pixel]])
    translated_img = cv2.warpAffine(img, M, (cols, rows))
    return translated_img

def rand_translate_by_factor(img: np.ndarray, translate_range_factor: float=0.1) -> np.ndarray:
    rows = img.shape[0]
    cols = img.shape[1]
    row_translate_range = max(int(rows*translate_range_factor),1)
    col_translate_range = max(int(cols*translate_range_factor),1)
    rand_rows = np.random.randint(low=-row_translate_range,high=row_translate_range)
    rand_cols = np.random.randint(low=-col_translate_range,high=col_translate_range)
    translated_img = translate_by_pixel(img, rand_rows, rand_cols)
    return translated_img

def rand_warp(img: np.ndarray, padding_bound: int=3, persp_bound: float=0.008, rot_bound: int=8, scale_diff_bound: float=0.2) -> np.ndarray:
    orig_row = img.shape[0]
    orig_col = img.shape[1]

    # random padding (make border if >0; do nothing if 0; randomly translate if <0)
    if padding_bound <= 0:
        c1, c2 = (0,0)
    else:
        c1, c2 = np.random.randint(low=-padding_bound, high=padding_bound, size=2)
    # random perspective params
    a, b = np.random.uniform(low=-persp_bound, high=persp_bound, size=2)
    # random rotation param
    theta = np.radians(np.random.randint(low=-rot_bound, high=rot_bound))
    # random scale difference param
    gamma = np.random.uniform(low=1-scale_diff_bound, high=1+scale_diff_bound)

    translation_matrix = np.array([[gamma, 0, 0],[0, 1, 0],[0, 0, 1]] , dtype=np.float32)
    perspective_matrix = np.array([[1, 0, 0],[0, 1, 0],[a, b, 1]], dtype=np.float32)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=np.float32)
    homography_matrix = translation_matrix.dot(perspective_matrix).dot(rotation_matrix)
    warpped_img = cv2.warpPerspective(img, homography_matrix, (orig_row*2, orig_col*2))
    warpped_img = crop_border(warpped_img)
    if padding_bound < 0:
        warpped_img = rand_translate_by_factor(warpped_img)
        warpped_img = crop_border(warpped_img)
    else:
        warpped_img = cv2.copyMakeBorder(warpped_img, -(c1*(c1<0)), c1*(c1>0), -(c2*(c2<0)), c2*(c2>0), cv2.BORDER_CONSTANT, 0)
    warpped_img = extend_canvas_square(warpped_img)
    warpped_img = cv2.resize(warpped_img, (orig_col, orig_row), interpolation = cv2.INTER_CUBIC)
    return warpped_img

def skeletonize(img):
    img1 = img.copy()
    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # Create an empty output image to hold values
    thin = np.zeros(img.shape, dtype='uint8')
    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(img1)!=0):
        # Erosion
        erode = cv2.erode(img1, kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset, thin)
        # Set the eroded image for next iteration
        img1 = erode.copy()
    return thin