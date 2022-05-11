import torch
from data_generator import StickerDataset
from torch.utils.data import DataLoader
from models import cnn
import datetime

batch_size = 32
epochs = 1000
device = "cuda:0"
datetime = datetime.datetime.now().strftime('%Y%m%d%H%M')

training_data = StickerDataset(img_dir="dataset/stickers_png/")
testing_data = StickerDataset(img_dir="dataset/stickers_png/")
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
max_label = testing_data.max_label()

model = cnn.cnn_feature_extraction(labels = max_label + 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
loss_fn = torch.nn.CrossEntropyLoss()

def train_feature_extraction():
    model = cnn.cnn_feature_extraction(labels = max_label + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch_i in range(epochs):
        print("Epoch:", epoch_i)
        n_correct = 0
        prev_best_loss = 10000
        prev_best_acc = 0
        for (train_features, train_labels) in train_dataloader:
            train_features = train_features.to(device)
            train_labels = train_labels.to(device)
            optimizer.zero_grad()
            logits = model(train_features)
            pred_probs = torch.nn.Softmax(dim=1)(logits)
            y_pred = pred_probs.argmax(1)
            loss = loss_fn(logits, train_labels)
            loss.backward()
            optimizer.step()
            batch_correct = sum(y_pred == train_labels)
            n_correct += batch_correct
        current_loss = loss.item()
        current_acc = (n_correct/len(training_data)).item()
        print("Training loss:", current_loss)
        print("Training acc:", current_acc)
        if (current_loss <= prev_best_loss) and (current_acc >= prev_best_acc):
            prev_best_loss = current_loss
            prev_best_acc = current_acc
            ckpt_path = "outputs/features_" + datetime + ".pt"
            torch.save(model.state_dict(), ckpt_path)
            print("Saved checkpoint at:", ckpt_path)


train_feature_extraction()