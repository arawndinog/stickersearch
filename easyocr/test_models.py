import torch
from craft import CRAFT
import cv2
import numpy as np

def remove_state_dict_prefix(state_dict):
    # remove "module." prefix from DDP/DP models for loading on non-DDP models
    state_dict = {k.partition('module.')[2]:state_dict[k] for k in state_dict.keys()}
    return state_dict

def main():
    device = "cpu"
    # ckpt_path = "../models/english_g2.pth"
    ckpt_path = "../models/craft_mlt_25k.pth"
    model = CRAFT()
    model_weights = torch.load(ckpt_path, map_location=torch.device(device))
    model_weights = remove_state_dict_prefix(model_weights)
    model.load_state_dict(model_weights)

    img_path = "../dataset/stickers_png/batch_3/cleaned/img_007.png"
    img = cv2.imread(img_path, 1)
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, 0)
    img = img/255.0
    img = torch.from_numpy(img)
    img = img.float()
     with torch.no_grad():
        y, _ = model(img)
    for out in y:
        # make score and link map
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys, mapper = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

main()  
