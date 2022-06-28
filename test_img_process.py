from utils import process_image, ftlib
import cv2
import numpy as np
import os
import torch
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

def test_pt_augmentation(img_list, out_dir):

    pt_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.8, 1), shear=None),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.Resize((128,128))
    ])

    result_img_list = []
    for i in range(len(img_list)):
        img = img_list[i]
        img = pt_transforms(img)
        img = img.numpy()
        img = img.squeeze()
        img = img * 255.0

        if out_dir:
            cv2.imwrite(out_dir + "test_" + str(i).zfill(2) + ".png", img)
        
        result_img_list.append(img)
    return result_img_list

def cv_mimic(img_orig):
    img = cv2.GaussianBlur(img_orig, (3,3), 1)

    # Morphological Gradient
    # kernel = np.ones((5,5),np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    # Canny
    img = cv2.Canny(img, 100, 200)

    # img = cv2.GaussianBlur(img, (5,5), 1)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    # img = (img > 100) * 255
    # img = img.astype(np.uint8)
    # kernel = np.ones((5,5), np.uint8)
    # img = cv2.dilate(img, kernel, iterations = 3)
    # img = cv2.erode(img, kernel, iterations = 2)
    # contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # img = cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=255, thickness=5)
    # img = cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=255, thickness=cv2.FILLED)
    # img = cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=255, thickness=4)
    # img = cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=255, thickness=-1)

    # img = cv2.resize(img, fx=0.5, fy=0.5, dsize=None)
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # kernel = np.ones((7,7), np.uint8)
    # img = cv2.dilate(img, kernel, iterations = 2)
    # img = cv2.erode(img, kernel, iterations = 2)
    
    # img = cv2.resize(img, fx=2, fy=2, dsize=None)
    img = remove_double_stroke(img)
    return img

def remove_double_stroke(img_orig):
    img = cv2.resize(img_orig, fx=0.5, fy=0.5, dsize=None, interpolation=cv2.INTER_AREA)
    kernel = np.ones((3,3), np.uint8)
    img = cv2.dilate(img, kernel, iterations = 1)
    img = cv2.GaussianBlur(img, (3,3), 1)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.resize(img, dsize=(img_orig.shape[1], img_orig.shape[0]), interpolation=cv2.INTER_CUBIC)
    # img = cv2.erode(img, kernel, iterations = 5)
    return img

def cv_stroke_thinning(img):
    kernel = np.ones((5,5), np.uint8)
    img = cv2.erode(img, kernel, iterations = 1)
    return img

def cv_warping(img_orig):
    img = process_image.rand_warp(img_orig, padding_bound=3, persp_bound=0.001, rot_bound=2, scale_diff_bound=0.2)
    return img

def test_augmentation_pipeline():
    # init photosketch
    device = "cuda"
    photosketch_ckpt_path = "models/photosketch_model_jit.pth"
    photosketch_model = torch.jit.load(photosketch_ckpt_path, map_location=torch.device(device))
    photosketch_model.eval()

    input_dir = "dataset/stickers_png/batch_3_cleaned/"
    output_dir = "outputs/processed_results7/"
    img_list = process_image.get_img_list(img_dir_path=input_dir, color_mode=0)
    for i in range(len(img_list)):
        img_orig = img_list[i]
        img = img_orig[:]
        # img = cv_mimic(img)
        
        # img = remove_double_stroke(img)

        # img = cv_stroke_thinning(img)

        # test photosketch
        img = 255 - img
        img = np.expand_dims(img, 2)
        img = np.repeat(img, 3, 2)
        img_th = transforms.ToTensor()(img)
        img_th = torch.unsqueeze(img_th, 0)
        img_th = img_th.to(device)
        img_th = img_th.float()
        img = photosketch_model(img_th)
        img = (img.cpu().detach().numpy()).squeeze()
        img = (img + 1) / 2.0 * 255.0
        img = 255 - img
        img = img.astype(np.uint8)

        img = remove_double_stroke(img)
        img = remove_double_stroke(img)

        # img = stroke_thinning(img)
        # img = 255-img
        # img = ftlib.fastThin(img)
        # img = 255-img
        # img = process_image.skeletonize(img)

        img_compare = np.concatenate((img_orig, img), axis=1)
        cv2.imwrite(output_dir + str(i).zfill(3) + ".png", img_compare)
        print(i)
    return



def test_augmentation_pipeline2():
    device = "cuda"
    hed_ckpt_path = "models/hed_model_gpu_jit.pth"
    hed_model = torch.jit.load(hed_ckpt_path, map_location=torch.device(device))
    hed_model.eval()

    input_dir = "dataset/stickers_png/batch_3_cleaned/"
    output_dir = "outputs/processed_results6/"
    img_list = process_image.get_img_list(img_dir_path=input_dir, color_mode=1)
    for i in range(len(img_list)):
        img_orig = img_list[i]
        img = img_orig[:]

        # prelim thinning
        kernel = np.ones((3,3), np.uint8)
        img = cv2.erode(img, kernel, iterations = 2)
        # hed

        # img = cv2.resize(img, dsize=(320, 320), interpolation=cv2.INTER_AREA)
        # img = process_image.extend_canvas(img, (320, 480))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float()
        img = img.to(device)

        hed_img = hed_model(img)
        
        img = hed_img.cpu().detach().numpy()
        img = img[0]
        img = img.transpose(1,2,0)
        img = img * 255.0
        img = img.astype(np.uint8)
        
        # binarize
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        

        # thinning
        img = 255-img
        img = ftlib.fastThin(img)
        img = 255-img

        img = cv2.GaussianBlur(img, (3,3), 1)
        kernel = np.ones((3,3), np.uint8)
        img = cv2.dilate(img, kernel, iterations = 1)
        img = remove_double_stroke(img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img = np.expand_dims(img, 2)
        img = np.repeat(img, 3, 2)

        img_compare = np.concatenate((img_orig, img), axis=1)
        cv2.imwrite(output_dir + str(i).zfill(3) + ".png", img_compare)
        print(i)
    return

def mimic_canny(input_dir, output_dir):
    print("Generating mimic_canny")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    img_list = process_image.get_img_list(img_dir_path=input_dir, color_mode=0)
    for i in range(len(img_list)):
        img_orig = img_list[i]
        img = img_orig[:]
        img = cv_mimic(img)      
        img = remove_double_stroke(img)
        img = cv_stroke_thinning(img)
        cv2.imwrite(output_dir + "img_" + str(i).zfill(3) + ".png", img)
    return

def mimic_photosketch(input_dir, output_dir):
    print("Generating mimic_photosketch")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    device = "cuda"
    photosketch_ckpt_path = "models/photosketch_model_jit.pth"
    photosketch_model = torch.jit.load(photosketch_ckpt_path, map_location=torch.device(device))
    photosketch_model.eval()

    img_list = process_image.get_img_list(img_dir_path=input_dir, color_mode=0)
    for i in range(len(img_list)):
        img_orig = img_list[i]
        img = img_orig[:]
        img = 255 - img
        img = np.expand_dims(img, 2)
        img = np.repeat(img, 3, 2)
        img_th = transforms.ToTensor()(img)
        img_th = torch.unsqueeze(img_th, 0)
        img_th = img_th.to(device)
        img_th = img_th.float()
        img = photosketch_model(img_th)
        img = (img.cpu().detach().numpy()).squeeze()
        img = (img + 1) / 2.0 * 255.0
        img = 255 - img
        img = img.astype(np.uint8)

        img = remove_double_stroke(img)
        img = remove_double_stroke(img)
        cv2.imwrite(output_dir + "img_" + str(i).zfill(3) + ".png", img)
    return

def mimic_hed(input_dir, output_dir):
    print("Generating mimic_hed")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    device = "cuda"
    hed_ckpt_path = "models/hed_model_gpu_jit.pth"
    hed_model = torch.jit.load(hed_ckpt_path, map_location=torch.device(device))
    hed_model.eval()

    img_list = process_image.get_img_list(img_dir_path=input_dir, color_mode=1)
    for i in range(len(img_list)):
        img_orig = img_list[i]
        img = img_orig[:]
        # prelim thinning
        kernel = np.ones((3,3), np.uint8)
        img = cv2.erode(img, kernel, iterations = 2)
        # hed
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float()
        img = img.to(device)

        hed_img = hed_model(img)
        
        img = hed_img.cpu().detach().numpy()
        img = img[0]
        img = img.transpose(1,2,0)
        img = img * 255.0
        img = img.astype(np.uint8)
        
        # binarize
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # thinning
        img = 255-img
        img = ftlib.fastThin(img)
        img = 255-img

        img = cv2.GaussianBlur(img, (3,3), 1)
        kernel = np.ones((3,3), np.uint8)
        img = cv2.dilate(img, kernel, iterations = 1)
        img = remove_double_stroke(img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img = np.expand_dims(img, 2)
        img = np.repeat(img, 3, 2)

        cv2.imwrite(output_dir + "img_" + str(i).zfill(3) + ".png", img)
    return

def test_aug():
    input_dir = "dataset/stickers_png/batch_1/hed/"
    output_dir = "outputs/processed_results8/"
    img_list = os.listdir(input_dir)
    for img_path in img_list:
        img = cv2.imread(input_dir + img_path, 0)
        img = process_image.rand_warp(img, padding_bound=3, persp_bound=0.001, rot_bound=2, scale_diff_bound=0.2)
        img = process_image.rand_rotate(img, 2)
        img = process_image.rand_translate_by_factor(img, 0.1)
        cv2.imwrite(output_dir + img_path, img)
    return

def gen_mimic():
    input_dir = "dataset/stickers_png/batch_2/"
    input_subdir = "cleaned/"

    mimic_canny(input_dir + input_subdir,       input_dir + "canny/")
    mimic_photosketch(input_dir + input_subdir, input_dir + "photosketch/")
    mimic_hed(input_dir + input_subdir,         input_dir + "hed/")


if __name__ == "__main__":
    test_aug()