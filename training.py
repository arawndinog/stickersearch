import torch
from utils.gen_data import StickerDataset, StickerDatasetTriplet
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from models import cnn, triplet_loss_pt
import datetime

device = "cuda:0"
datetime = datetime.datetime.now().strftime('%Y%m%d%H%M')

def train_feature_extraction():
    batch_size = 16
    # input_size = 128
    input_size = 224

    img_dir_path_list = ["dataset/stickers_png/batch_1/cleaned/", "dataset/stickers_png/batch_3/cleaned/"]
    training_data = StickerDataset(img_dir_list=img_dir_path_list, input_size=input_size, augmentation=False)
    testing_data = StickerDataset(img_dir_list=img_dir_path_list, input_size=input_size, augmentation=False)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=False)
    max_label = testing_data.max_label()

    # seed_ckpt_path = "outputs/features_202205311834.pt"
    # seed_ckpt_path = "outputs/features_202206010133.pt"
    # seed_ckpt_path = "outputs/features_202206010244.pt"
    seed_ckpt_path = ""

    # model = cnn.cnn1(labels = max_label + 1)
    # model = cnn.cnn2(labels = max_label + 1)
    model = torchvision.models.convnext_small(pretrained=False, num_classes=max_label + 1)
    model._modules["features"][0][0] = torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
    model.to(device)
    model.train()

    if seed_ckpt_path:
        model.load_state_dict(torch.load(seed_ckpt_path, map_location=torch.device(device)))
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_i = 0
    while True:
        print("Epoch:", epoch_i)
        n_correct = 0
        loss_list = []
        prev_best_loss = 10000
        prev_best_acc = 0
        for (train_features, train_labels, _) in train_dataloader:
            train_features = train_features.to(device)
            train_labels = train_labels.to(device)
            optimizer.zero_grad()
            logits = model(train_features)
            pred_probs = torch.nn.Softmax(dim=1)(logits)
            y_pred = pred_probs.argmax(1)
            loss = loss_fn(logits, train_labels)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            batch_correct = sum(y_pred == train_labels)
            n_correct += batch_correct
        current_loss = sum(loss_list)/len(loss_list)
        current_acc = (n_correct/len(training_data)).item()
        print("Training loss:", current_loss)
        print("Training acc:", current_acc)
        if (epoch_i % 10 == 0) and (current_loss <= prev_best_loss) and (current_acc >= prev_best_acc):
            prev_best_loss = current_loss
            prev_best_acc = current_acc
            ckpt_path = "outputs/features_" + datetime + ".pt"
            torch.save(model.state_dict(), ckpt_path)
            print("Saved checkpoint at:", ckpt_path)
        epoch_i += 1

def train_similarity():
    batch_size = 16
    # input_size = 128
    input_size = 224

    img_dir_path_list = ["dataset/stickers_png/batch_1/", "dataset/stickers_png/batch_3/"]
    training_data = StickerDatasetTriplet(img_dir_list=img_dir_path_list, input_size=input_size)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    max_label = training_data.max_label()

    # seed_ckpt_path = ""
    # seed_ckpt_path = "outputs/features_202206192207.pt"
    # seed_ckpt_path = "outputs/features_202206192243.pt"      # best tri2 cnn2 + aug
    # seed_ckpt_path = "outputs/features_202206192311.pt"      # best tri2 cnn2 + aug
    # seed_ckpt_path = "outputs/features_202206230039.pt"      # best tri2 cnn2 + aug
    # seed_ckpt_path = "outputs/features_202206280113.pt"      # cnn2 triplet pairwise mimic
    seed_ckpt_path = "outputs/features_202206280114.pt"      # convnext triplet pairwise mimic

    # model = cnn.cnn1(labels = max_label + 1)
    # model = cnn.cnn2(labels = max_label + 1)
    model = torchvision.models.convnext_small(pretrained=False, num_classes=max_label + 1)
    model._modules["features"][0][0] = torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
    model.to(device)
    if seed_ckpt_path:
        model.load_state_dict(torch.load(seed_ckpt_path, map_location=torch.device(device)))
        
    # feature_out = create_feature_extractor(model, {'fc2':"features_layer"})
    feature_out = create_feature_extractor(model, {'classifier.2':'features_layer'})
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    # loss_fn = torch.nn.TripletMarginLoss(margin=100)
    # loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.CosineSimilarity(dim=1), margin=10)
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance(), margin=100)

    epoch_i = 0
    while True:
        print("Epoch:", epoch_i)
        n_correct = 0
        loss_list = []
        prev_best_loss = 10000
        prev_best_acc = 0
        for (train_features_anc, train_features_pos, train_features_neg, train_labels_pos, train_labels_neg) in train_dataloader:
            train_features = torch.concat((train_features_anc, train_features_pos, train_features_neg), 0)
            train_labels = torch.concat((train_labels_pos, train_labels_pos, train_labels_neg), 0)
            train_features = train_features.to(device)
            train_labels = train_labels.to(device)
            optimizer.zero_grad()
            features = feature_out(train_features)['features_layer']
            features_anc = features[:train_features_anc.shape[0]]
            features_pos = features[train_features_anc.shape[0]:train_features_anc.shape[0] + train_features_pos.shape[0]]
            features_neg = features[train_features_anc.shape[0] + train_features_pos.shape[0]:]
            # loss, _ = triplet_loss_pt.batch_all_triplet_loss(train_labels, features, margin=100, squared=False)
            loss = loss_fn(features_anc, features_pos, features_neg)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
        current_loss = sum(loss_list)/len(loss_list)
        print("Training loss:", current_loss)
        if (epoch_i % 100 == 0) and (current_loss <= prev_best_loss): # and (current_acc >= prev_best_acc):
            prev_best_loss = current_loss
            # prev_best_acc = current_acc
            ckpt_path = "outputs/features_" + datetime + ".pt"
            torch.save(model.state_dict(), ckpt_path)
            print("Saved checkpoint at:", ckpt_path)
        epoch_i += 1


train_similarity()
