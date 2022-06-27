import torch
import torchvision
import cv2
import numpy as np
from utils import process_image

def inference():
    ckpt_path = "models/photosketch_model_jit.pth"
    # ckpt_path = "models/model_jit.pth"
    # test_img_dir = "dataset/stickers_png/batch_1_cleaned/"
    test_img_dir = "outputs/processed_results4/"
    # test_img_dir = "/home/adrian/Repositories/PhotoSketch/examples/"
    # output_dir = "outputs/photosketch_result1/"
    output_dir = "outputs/photosketch_result2/"
    img_list = process_image.get_img_list(test_img_dir, 1)
    
    model = torch.jit.load(ckpt_path, map_location=torch.device("cpu"))
    model.eval()
    for i in range(len(img_list)):
        img = img_list[i]
        # img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
        img = 255-img
        img = torchvision.transforms.ToTensor()(img)
        # img = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        img = img.unsqueeze(0)
        print(img.shape)
        exit()
        output = model(img)
        result = (output.detach().numpy()).squeeze()
        result = (result + 1) / 2.0 * 255.0
        result = 255 - result
        result = result.astype(np.uint8)
        cv2.imwrite(output_dir + "test_" + str(i).zfill(3) + ".png", result)
        print(i)
    return

inference()