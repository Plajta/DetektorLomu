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