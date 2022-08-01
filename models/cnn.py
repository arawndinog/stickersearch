import torch

class cnn1(torch.nn.Module):
    # expect input 64x64
    def __init__(self, labels):
        super(cnn1, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 50, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(50, 100, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(100, 150, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(150, 200, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(200, 250, 3, padding=1)
        self.conv6 = torch.nn.Conv2d(250, 300, 3, padding=1)
        self.conv7 = torch.nn.Conv2d(300, 350, 3, padding=1)
        self.conv8 = torch.nn.Conv2d(350, 400, 3, padding=1)
        self.conv9 = torch.nn.Conv2d(400, 450, 3, padding=1)
        self.conv10 = torch.nn.Conv2d(450, 500, 3, padding=1)

        self.max_pool1 = torch.nn.MaxPool2d(2, stride=(2, 2))

        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(p=0.1)
        self.drop2 = torch.nn.Dropout(p=0.2)
        self.drop3 = torch.nn.Dropout(p=0.3)
        self.drop4 = torch.nn.Dropout(p=0.4)
        self.drop5 = torch.nn.Dropout(p=0.5)

        self.fc1 = torch.nn.Linear(500 * 2 * 2, 900)
        self.fc2 = torch.nn.Linear(900, 100)
        self.fc3 = torch.nn.Linear(100, labels)

    def forward(self, x):
        # conv_layer1
        x = self.conv1(x)       # 64x64
        x = self.relu(x)
        # conv_layer2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.max_pool1(x)   # 32x32
        # conv_layer3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.drop1(x)
        # conv_layer4
        x = self.conv4(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.max_pool1(x)   # 16x16
        # conv_layer5
        x = self.conv5(x)
        x = self.relu(x)
        x = self.drop2(x)
        # conv_layer6
        x = self.conv6(x)
        x = self.relu(x)
        x = self.drop3(x)
        x = self.max_pool1(x)   # 8x8
        # conv_layer7
        x = self.conv7(x)
        x = self.relu(x)
        x = self.drop3(x)
        # conv_layer8
        x = self.conv8(x)
        x = self.relu(x)
        x = self.drop4(x)
        x = self.max_pool1(x)   # 4x4
        # conv_layer9
        x = self.conv9(x)
        x = self.relu(x)
        x = self.drop4(x)
        # conv_layer10
        x = self.conv10(x)
        x = self.relu(x)
        x = self.drop4(x)
        x = self.max_pool1(x)   # 2x2

        # fc_layer1
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop5(x)
        # fc_layer2
        x = self.fc2(x)
        x = self.relu(x)
        # output_layer
        x = self.fc3(x)

        return x

class cnn2(torch.nn.Module):
    # expect input 128x128
    def __init__(self, labels):
        super(cnn2, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 50, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(50, 100, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(100, 150, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(150, 200, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(200, 250, 3, padding=1)
        self.conv6 = torch.nn.Conv2d(250, 300, 3, padding=1)
        self.conv7 = torch.nn.Conv2d(300, 350, 3, padding=1)
        self.conv8 = torch.nn.Conv2d(350, 400, 3, padding=1)
        self.conv9 = torch.nn.Conv2d(400, 450, 3, padding=1)
        self.conv10 = torch.nn.Conv2d(450, 500, 3, padding=1)
        self.conv11 = torch.nn.Conv2d(500, 550, 3, padding=1)
        self.conv12 = torch.nn.Conv2d(550, 600, 3, padding=1)

        self.max_pool1 = torch.nn.MaxPool2d(2, stride=(2, 2))

        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(p=0.1)
        self.drop2 = torch.nn.Dropout(p=0.2)
        self.drop3 = torch.nn.Dropout(p=0.3)
        self.drop4 = torch.nn.Dropout(p=0.4)
        self.drop5 = torch.nn.Dropout(p=0.5)

        self.fc1 = torch.nn.Linear(600 * 2 * 2, 900)
        self.fc2 = torch.nn.Linear(900, 100)
        self.fc3 = torch.nn.Linear(100, labels)

    def forward(self, x):
        # conv_layer1
        x = self.conv1(x)       # 128x128
        x = self.relu(x)
        # conv_layer2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.max_pool1(x)   # 64x64
        # conv_layer3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.drop1(x)
        # conv_layer4
        x = self.conv4(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.max_pool1(x)   # 32x32
        # conv_layer5
        x = self.conv5(x)
        x = self.relu(x)
        x = self.drop2(x)
        # conv_layer6
        x = self.conv6(x)
        x = self.relu(x)
        x = self.drop3(x)
        x = self.max_pool1(x)   # 16x16
        # conv_layer7
        x = self.conv7(x)
        x = self.relu(x)
        x = self.drop3(x)
        # conv_layer8
        x = self.conv8(x)
        x = self.relu(x)
        x = self.drop4(x)
        x = self.max_pool1(x)   # 8x8
        # conv_layer9
        x = self.conv9(x)
        x = self.relu(x)
        x = self.drop4(x)
        # conv_layer10
        x = self.conv10(x)
        x = self.relu(x)
        x = self.drop4(x)
        x = self.max_pool1(x)   # 4x4
        # conv_layer11
        x = self.conv11(x)
        x = self.relu(x)
        x = self.drop4(x)
        # conv_layer12
        x = self.conv12(x)
        x = self.relu(x)
        x = self.drop4(x)
        x = self.max_pool1(x)   # 2x2

        # fc_layer1
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop5(x)
        # fc_layer2
        x = self.fc2(x)
        x = self.relu(x)
        # output_layer
        x = self.fc3(x)

        return x


class cnn2_deploy(torch.nn.Module):
    # expect input 128x128
    def __init__(self):
        super(cnn2_deploy, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 50, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(50, 100, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(100, 150, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(150, 200, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(200, 250, 3, padding=1)
        self.conv6 = torch.nn.Conv2d(250, 300, 3, padding=1)
        self.conv7 = torch.nn.Conv2d(300, 350, 3, padding=1)
        self.conv8 = torch.nn.Conv2d(350, 400, 3, padding=1)
        self.conv9 = torch.nn.Conv2d(400, 450, 3, padding=1)
        self.conv10 = torch.nn.Conv2d(450, 500, 3, padding=1)
        self.conv11 = torch.nn.Conv2d(500, 550, 3, padding=1)
        self.conv12 = torch.nn.Conv2d(550, 600, 3, padding=1)

        self.max_pool1 = torch.nn.MaxPool2d(2, stride=(2, 2))

        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(p=0.1)
        self.drop2 = torch.nn.Dropout(p=0.2)
        self.drop3 = torch.nn.Dropout(p=0.3)
        self.drop4 = torch.nn.Dropout(p=0.4)
        self.drop5 = torch.nn.Dropout(p=0.5)

        self.fc1 = torch.nn.Linear(600 * 2 * 2, 900)
        self.fc2 = torch.nn.Linear(900, 100)
        self.fc3 = torch.nn.Linear(100, 271)

    def forward(self, x):
        # conv_layer1
        x = self.conv1(x)       # 128x128
        x = self.relu(x)
        # conv_layer2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.max_pool1(x)   # 64x64
        # conv_layer3
        x = self.conv3(x)
        x = self.relu(x)
        x = self.drop1(x)
        # conv_layer4
        x = self.conv4(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.max_pool1(x)   # 32x32
        # conv_layer5
        x = self.conv5(x)
        x = self.relu(x)
        x = self.drop2(x)
        # conv_layer6
        x = self.conv6(x)
        x = self.relu(x)
        x = self.drop3(x)
        x = self.max_pool1(x)   # 16x16
        # conv_layer7
        x = self.conv7(x)
        x = self.relu(x)
        x = self.drop3(x)
        # conv_layer8
        x = self.conv8(x)
        x = self.relu(x)
        x = self.drop4(x)
        x = self.max_pool1(x)   # 8x8
        # conv_layer9
        x = self.conv9(x)
        x = self.relu(x)
        x = self.drop4(x)
        # conv_layer10
        x = self.conv10(x)
        x = self.relu(x)
        x = self.drop4(x)
        x = self.max_pool1(x)   # 4x4
        # conv_layer11
        x = self.conv11(x)
        x = self.relu(x)
        x = self.drop4(x)
        # conv_layer12
        x = self.conv12(x)
        x = self.relu(x)
        x = self.drop4(x)
        x = self.max_pool1(x)   # 2x2

        # fc_layer1
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop5(x)
        # fc_layer2
        x1 = self.fc2(x)
        x = self.relu(x1)
        # output_layer
        x = self.fc3(x)

        return x1