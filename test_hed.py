import torch
import torchvision
import cv2
import numpy as np
from utils import process_image, ftlib

def inference():
    device = "cuda"
    ckpt_path = "models/hed_model_gpu_jit.pth"
    test_img_dir = "dataset/stickers_png/batch_3_cleaned/"
    output_dir = "outputs/hed_result1/"
    img_list = process_image.get_img_list(test_img_dir, 1)
    
    model = torch.jit.load(ckpt_path, map_location=torch.device(device))
    model.eval()
    for i in range(len(img_list)):
        img = img_list[i]
        kernel = np.ones((5,5), np.uint8)
        img = cv2.erode(img, kernel, iterations = 1)
        # img = np.expand_dims(img, 2)
        # img = np.repeat(img, 3, 2)
        # img = img[:, :, ::-1]
        # img = cv2.resize(img, dsize=(320, 320), interpolation=cv2.INTER_AREA)
        # img = process_image.extend_canvas(img, (320, 480))
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float()
        img = img.to(device)

        result = model(img)
        
        result = result.cpu().detach().numpy()
        result = result[0]
        result = result.transpose(1,2,0)
        result = result*255.0
        result = result.astype(np.uint8)

        # result = 255-result
        # result = ftlib.fastThin(result)
        # result = 255-result

        cv2.imwrite(output_dir + "test_" + str(i).zfill(3) + ".png", result)
        print(i)
    return

inference()