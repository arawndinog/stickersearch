import torch
from utils.gen_data import StickerDataset, StickerDatasetTriplet
from utils import process_image
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from models import cnn
import numpy as np
import cv2
from torch.utils.mobile_optimizer import optimize_for_mobile

batch_size = 1
# input_size = 64
# input_size = 128
input_size = 224
device = "cuda"

# testing_data = StickerDataset(img_dir_list=["dataset/stickers_png/batch_2_cleaned/"], input_size=input_size, augmentation=True)
# testing_data = StickerDataset(img_dir_list=["dataset/stickers_png/batch_1_cleaned/"], input_size=input_size, augmentation=True)
testing_data = StickerDatasetTriplet(img_dir_list=["dataset/stickers_png/batch_2/"], input_size=input_size, augment=False)
testing_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)
max_label = testing_data.max_label()
custom_testing_data = StickerDataset(img_dir_list=["dataset/realworld/"], input_size=input_size, augmentation=False)
custom_testing_dataloader = DataLoader(custom_testing_data, batch_size=batch_size, shuffle=False)

def test_imgs():
    ckpt_path = ""
    # ckpt_path = "outputs/checkpoints/features_202205311834.pt"    # best pre bg fix
    # ckpt_path = "outputs/checkpoints/features_202206010056.pt"      # best post bg fix
    # ckpt_path = "outputs/checkpoints/features_202206010154.pt"      # best post aug
    # ckpt_path = "outputs/checkpoints/features_202206010244.pt"      # best post cnn2 + aug
    # ckpt_path = "outputs/checkpoints/features_202206010329.pt"      # convnext + aug
    # ckpt_path = "outputs/checkpoints/features_202206192200.pt"      # best tri cnn2 + aug
    # ckpt_path = "outputs/checkpoints/features_202206192243.pt"      # best tri2 cnn2 + aug
    # ckpt_path = "outputs/checkpoints/features_202206230039.pt"      # best tri3 cnn2 + aug
    # ckpt_path = "outputs/checkpoints/features_202206231757.pt"      # best tri_pariwise1 cnn2 + aug
    # ckpt_path = "outputs/checkpoints/features_202206280113.pt"      # cnn2 triplet pairwise mimic
    # ckpt_path = "outputs/checkpoints/features_202206280114.pt"      # convnext triplet pairwise mimic
    # ckpt_path = "outputs/checkpoints/features_202206282334.pt"      # cnn2 triplet pairwise mimic + warp aug
    # ckpt_path = "outputs/checkpoints/features_202206282339.pt"      # convnext triplet pairwise mimic + warp aug
    # ckpt_path = "outputs/checkpoints/features_202206300021a.pt"      # cnn2 triplet pairwise mimic + elastic aug
    # ckpt_path = "outputs/checkpoints/features_202206300018a.pt"      # convnext triplet pairwise mimic + elastic aug
    # ckpt_path = "outputs/checkpoints/features_202206300021b.pt"      # cnn2 triplet pairwise mimic + elastic aug B
    ckpt_path = "outputs/checkpoints/features_202206300018b.pt"      # convnext triplet pairwise mimic + elastic aug B
    
    
    # labels = 117
    # labels = 154
    labels = 271
    
    # model = cnn.cnn1(labels = labels)
    # model = cnn.cnn2(labels = labels)
    # model = cnn.cnn2_deploy()
    model = torchvision.models.convnext_small(pretrained=False, num_classes=labels)
    model._modules["features"][0][0] = torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()

    # train_nodes, eval_nodes = get_graph_node_names(model)
    # print(train_nodes)
    # exit()

    # feature_out = create_feature_extractor(model, {'fc2':"features_layer"})
    feature_out = create_feature_extractor(model, {'classifier.2':'features_layer'})
    cos = torch.nn.CosineSimilarity(dim=1)

    features_list = []
    label_to_path_dict = dict()
    for (test_data, _, _, test_label, _, test_data_path) in testing_dataloader:
        test_data = test_data.to(device)

        
        # traced_script_module = torch.jit.trace(model, test_data)
        # traced_script_module_optimized = optimize_for_mobile(traced_script_module)
        # traced_script_module_optimized._save_for_lite_interpreter("models/mobile_model.ptl")
        # exit()

        # test classification accuracy, only for training dataset
        # logits = model(test_data)
        # pred_probs = torch.nn.Softmax(dim=1)(logits)
        # y_pred = pred_probs.argmax(1)
        # print(y_pred == test_labels)

        features = feature_out(test_data)['features_layer']
        features_list.append(features.detach())
        label_to_path_dict[str(test_label.item())] = test_data_path[0]
    features_arr = torch.cat(features_list, 0)
    # features_arr_np = features_arr.cpu().detach().numpy()
    # print(features_arr_np.shape)
    # features_arr_np = features_arr_np.astype(">i2")
    # features_arr_np.tofile("models/features_db.bin")
    # exit()
    # torch.save(features_arr, "models/features_db.pt")
    # exit()

    cos_n_correct = 0
    dist_n_correct = 0
    cos_top5_n_correct = 0
    dists_top5_n_correct = 0
    for (_, test_data, _, test_label, _, _) in testing_dataloader:
        # temp = test_data[0][0].detach().numpy()
        # temp = (temp*255.0).astype(np.uint8)
        # process_image.preview_img(temp)
        # exit()
        test_data = test_data.to(device)
        test_label = test_label.to(device)
        features = feature_out(test_data)['features_layer']
        features = features.detach()

        cos_sim = cos(features, features_arr)
        dist = torch.sum(torch.sub(features, features_arr) ** 2, dim=1)
        # print(dist[0])

        cos_preds = torch.argsort(cos_sim, descending=True)[:5]
        dist_preds = torch.argsort(dist, descending=False)[:5]

        cos_n_correct += (test_label == cos_preds[0])
        dist_n_correct += (test_label == dist_preds[0])
        cos_top5_n_correct += (test_label in cos_preds)
        dists_top5_n_correct += (test_label in dist_preds)
    cos_acc = (cos_n_correct/len(testing_data)).item()
    dist_acc = (dist_n_correct/len(testing_data)).item()
    cos_top5_acc = cos_top5_n_correct/len(testing_data)
    dist_top5_acc = dists_top5_n_correct/len(testing_data)
    print(cos_acc)
    print(dist_acc)
    print(cos_top5_acc)
    print(dist_top5_acc)

    
    for (test_data, _, test_data_path) in custom_testing_dataloader:
        test_data = test_data.to(device)
        features = feature_out(test_data)['features_layer']
        features = features.detach()

        cos_sim = cos(features, features_arr)
        dist = torch.sum(torch.sub(features, features_arr) ** 2, dim=1)

        cos_preds = torch.argsort(cos_sim, descending=True)[:5]
        dist_preds = torch.argsort(dist, descending=False)[:5]

        print(test_data_path)
        print(cos_preds)
        print(dist_preds)

    return

test_imgs()