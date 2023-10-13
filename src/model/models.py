import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo

import Wandb
import os

#init
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LOG = False
SAVE = True
BATCH_SIZE = 8
BATCH_SIZE_TEST = 1

class Universal(nn.Module):
    def __init__(self) -> None:
        super(Universal, self).__init__()
        self.config = {
            "epochs": 10,
            "metric": "accuracy",
            "batch-size": BATCH_SIZE,
            "dropout": 0.2
        }
        self.model_iter = 0

    def train_net(self, train):
        #variables
        total_loss = 0
        total_correct = 0
        idx = 0

        self.train()
        for X, y in train:
            self.optimizer.zero_grad()

            X = X.to(DEVICE)
            y = y.to(DEVICE)
            y = torch.unsqueeze(y, 1).float()

            output = self(X)
            output = output.to(DEVICE)

            #loss = F.binary_cross_entropy(output, y)
            loss = F.binary_cross_entropy(output, y)
            loss.backward()

            self.optimizer.step()

            total_loss += loss.detach().item()

            for i in range(output.shape[0]):
                if torch.round(output[i]) == y[i]:
                    total_correct += 1

            idx += 1
        if LOG:
            Wandb.wandb.log({"train/loss": total_loss / idx})
        
        print(f"train loss: {total_loss / idx}, test acc: {total_correct / (idx * BATCH_SIZE)}")


    def test_net(self, test):
        #variables
        total_loss = 0
        total_correct = 0
        idx = 0
        
        self.eval()
        with torch.no_grad():
            for X, y in test:
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                y = torch.unsqueeze(y, 1).float()

                output = self(X)
                output = output.to(DEVICE)

                #loss = F.binary_cross_entropy(output, y)
                loss = F.binary_cross_entropy(output, y)
                total_loss += loss.detach().item()

                for i in range(output.shape[0]):
                    if torch.round(output[i]) == y[i]:
                        total_correct += 1

                idx += 1
        if LOG:
            #log every data
            Wandb.wandb.log({"test/loss": total_loss / idx})

        print(f"test loss: {total_loss / idx}, test acc: {total_correct / (idx * BATCH_SIZE)}")

    def run(self, train, test):
        print(f"model DEVICE")
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        print(f"SUMMARY:")
        torchinfo.summary(self, (1, 1, 640, 480))

        self.to(DEVICE)

        if LOG:
            Wandb.Init("Lomy", configuration=self.config, run_name=self.model_config["name"])

        self.test_net(test)
        for i_epoch in range(self.config["epochs"]):
            print(f"epoch {i_epoch}")

            #train net
            self.train_net(train)

            #test net
            self.test_net(test)

            if SAVE:
                torch.save(self, f"{os.getcwd()}/src/model/saved/{self.model_config['name']}{self.model_iter}.pth")

            self.model_iter += 1

        if LOG:
            Wandb.End()

    def inference(self, img):
        img = img.to(DEVICE)

        output = self(img)
        output = output.to(DEVICE)

        if output.detach().item() == 0:
            print("detekovano: stepny lom")
        elif output.detach().item() == 0:
            print("detekovano: tvarny lom")

class NeuralNet(Universal):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.model_config = {
            "name": "NeuralNet"
        }

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

        self.model_config = {
            "name": "ConvNeuralNet"
        }

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

        x = self.drop3(self.layer3(x))
        x = self.activation_l3(x)

        x = self.drop4(self.layer4(x))
        x = self.activation_l4(x)

        return self.activation_fin(self.layer5(x))