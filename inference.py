import torch
from utils.gen_data import StickerDataset
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from models import cnn
import numpy as np
import cv2

batch_size = 1
# input_size = 64
input_size = 128
# input_size = 224
device = "cuda"

testing_data = StickerDataset(img_dir_list=["dataset/stickers_png/batch_2_cleaned/"], input_size=input_size, augmentation=True)
# testing_data = StickerDataset(img_dir_list=["dataset/stickers_png/batch_1_cleaned/"], input_size=input_size, augmentation=True)
testing_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)
max_label = testing_data.max_label()

def test_imgs_cosine():
    # ckpt_path = "outputs/features_202205311834.pt"    # best pre bg fix
    # ckpt_path = "outputs/features_202206010056.pt"      # best post bg fix
    # ckpt_path = "outputs/features_202206010154.pt"      # best post aug
    # ckpt_path = "outputs/features_202206010244.pt"      # best post cnn2 + aug
    ckpt_path = "outputs/features_202206010329.pt"      # convnext + aug
    # ckpt_path = "outputs/features_202206192200.pt"      # best tri cnn2 + aug
    # ckpt_path = "outputs/features_202206192243.pt"      # best tri2 cnn2 + aug
    # ckpt_path = "outputs/features_202206230039.pt"      # best tri3 cnn2 + aug
    # ckpt_path = "outputs/features_202206231757.pt"      # best tri_pariwise1 cnn2 + aug
    
    
    labels = 117
    # labels = 154
    # labels = 271
    
    # model = cnn.cnn1(labels = labels)
    # model = cnn.cnn2(labels = labels)
    model = torchvision.models.convnext_small(pretrained=False, num_classes=labels)
    model._modules["features"][0][0] = torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    # model.to(device)
    model.eval()

    # train_nodes, eval_nodes = get_graph_node_names(model)
    # print(train_nodes)
    # exit()

    # feature_out = create_feature_extractor(model, {'fc2':"features_layer"})
    feature_out = create_feature_extractor(model, {'classifier.2':'features_layer'})
    cos = torch.nn.CosineSimilarity(dim=1)

    features_list = []
    test_data_paths_list = []
    for (test_data, test_labels, test_data_paths) in testing_dataloader:
        test_data = test_data.to(device)

        # test classification accuracy, only for training dataset
        # logits = model(test_data)
        # pred_probs = torch.nn.Softmax(dim=1)(logits)
        # y_pred = pred_probs.argmax(1)
        # print(y_pred == test_labels)

        features = feature_out(test_data)['features_layer']
        features_list.append(features)
        test_data_paths_list += test_data_paths
    features_arr = torch.cat(features_list, 0)

    n_correct = 0
    for (test_data, test_label, test_data_path) in testing_dataloader:
        test_data = test_data.to(device)
        features = feature_out(test_data)['features_layer']
        cos_sim = cos(features, features_arr)
        pred = cos_sim.argmax()
        n_correct += (pred == test_label)
    cos_acc = (n_correct/len(testing_data)).item()
    print(cos_acc)

    n_correct = 0
    for (test_data, test_label, test_data_path) in testing_dataloader:
        test_data = test_data.to(device)
        features = feature_out(test_data)['features_layer']
        dist = torch.sum(torch.sub(features, features_arr) ** 2, dim=1)
        pred = dist.argmin(0)
        n_correct += (pred == test_label)
    dist_acc = (n_correct/len(testing_data)).item()
    print(dist_acc)

    return

test_imgs_cosine()