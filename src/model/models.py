import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#init
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Universal(nn.Module):
    def __init__(self) -> None:
        super(Universal, self).__init__()
        self.config = {
            "epochs": 10,
            "optimizer": "adam",
            "metric": "accuracy",
            "batch-size": 8,
            "dropout": 0.2
        }

    def train_net(self, train):
        pass

    def test_net(self, test):
        pass

    def run(self, train, test):

        print(f"model DEVICE")

class NeuralNet(Universal):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.flatten = nn.Flatten()

        self.layer1 = nn.Linear(307200, 65536)
        self.drop1 = nn.Dropout(self.config["dropout"])
        self.activation1 = nn.ReLU()

        self.layer2 = nn.Linear(65536, 16384)
        self.drop2 = nn.Dropout(self.config["dropout"])
        self.activation2 = nn.ReLU()

        self.layer3 = nn.Linear(16384, 4096)
        self.drop3 = nn.Dropout(self.config["dropout"])
        self.activation3 = nn.ReLU()

        self.layer4 = nn.Linear(4096, 256)
        self.drop4 = nn.Dropout(self.config["dropout"])
        self.activation4 = nn.ReLU()

        self.layer5 = nn.Linear(256, 16)
        self.drop5 = nn.Dropout(self.config["dropout"])
        self.activation5 = nn.ReLU()

        self.layer6 = nn.Linear(16, 1)
        self.activation_fin = nn.Sigmoid()
    
    def forward(self, x):
        x = self.drop1(self.layer1(x))
        x = self.activation1(x)

        x = self.drop2(self.layer2(x))
        x = self.activation2(x)

        x = self.drop3(self.layer3(x))
        x = self.activation3(x)

        x = self.drop4(self.layer4(x))
        x = self.activation4(x)

        x = self.drop5(self.layer5(x))
        x = self.activation5(x)

        x = self.activation_fin(self.layer6(x))
        return x

class ConvNeuralNet(Universal):
    def __init__(self):
        super(ConvNeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv2d(3, 6, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv2d(6, 16, (3, 3))
        self.pool3 = nn.MaxPool2d((2, 2))
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv2d(16, 32, (3, 3))
        self.pool4 = nn.MaxPool2d((2, 2))
        self.activation4 = nn.ReLU()

        self.conv5 = nn.Conv2d(32, 32, (3, 3))
        self.pool5 = nn.AvgPool2d((2, 2))
        self.activation5 = nn.ReLU()

        self.flat = nn.Flatten()

        self.layer1 = nn.Linear(7488, 2048)
        self.activation_l1 = nn.ReLU()

        self.layer2 = nn.Linear(2048, 512)
        self.activation_l2 = nn.ReLU()

        self.layer3 = nn.Linear(512, 128)
        self.drop3 = nn.Dropout(self.config["dropout"])
        self.activation_l3 = nn.ReLU()

        self.layer4 = nn.Linear(128, 32)
        self.drop4 = nn.Dropout(self.config["dropout"])
        self.activation_l4 = nn.ReLU()

        self.layer5 = nn.Linear(32, 1)
        self.activation_fin = nn.Sigmoid()

    def forward(self, x):
        #conv
        x = self.conv1(x)
        x = self.activation1(self.pool1(x))

        x = self.conv2(x)
        x = self.activation2(self.pool2(x))

        x = self.conv3(x)
        x = self.activation3(self.pool3(x))

        x = self.conv4(x)
        x = self.activation4(self.pool4(x))

        x = self.conv5(x)
        x = self.activation5(self.pool5(x))

        x = self.flat(x)

        #linear
        x = self.activation_l1(self.layer1(x))

        x = self.activation_l2(self.layer2(x))

        x = self.drop3(self.layer4(x))
        x = self.activation_l3(x)

        x = self.drop4(self.layer4(x))
        x = self.activation_l4(x)

        x = self.activation_fin(self.layer5(x))

        return x