import torch
from data_generator import StickerDataset
from torch.utils.data import DataLoader
from models import cnn

batch_size = 8
epochs = 10

training_data = StickerDataset(img_dir="dataset/stickers_png/")
testing_data = StickerDataset(img_dir="dataset/stickers_png/")
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)
max_label = testing_data.max_label()

model = cnn.cnn_feature_extraction(labels = max_label + 1)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = torch.nn.CrossEntropyLoss()

def train_feature_extraction():
    for epoch_i in range(epochs):
        for (train_features, train_labels) in train_dataloader:
            print(train_features.shape)
            print(train_labels)
            optimizer.zero_grad()
            output = model(train_features)
            loss = loss_fn(output, train_labels)
            loss.backward()
            optimizer.step()
        print(loss)

train_feature_extraction()