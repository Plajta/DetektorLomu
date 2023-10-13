from torch.utils.data import Dataset, DataLoader, random_split

from models import BATCH_SIZE, ConvNeuralNet
from process import Loader
import os
import cv2
import torch
import numpy as np

import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

class LomyDataset(Dataset):
    def __init__(self):
        print(os.getcwd())
        self.loader = Loader("dataset/lomy/stepnylom_jpg", "dataset/lomy/tvarnylom_jpg")
    
    def __getitem__(self, idx):
        data = self.loader.get(idx).copy()
        return ImgTransform(data[0], "torch"), torch.tensor(data[1]) #X, y
    
    def __len__(self):
        return self.loader.get_length()
    
def ImgTransform(img, convert_to):
    if convert_to == "numpy": #to convert to numpy array
        img = torch.squeeze(img, 0)
        img_np = img.detach().numpy() * 255
        return img_np.astype("uint8")
    elif convert_to == "torch":
        return transform(img)
    
def inspect_dataset(index, dataloader):
    img = dataloader.dataset[index][0]
    label = dataloader.dataset[index][1]

    img_np = ImgTransform(img, "numpy")

    cv2.imshow("test", img_np)
    print(label)
    cv2.waitKey(0)

def inspect_classes(dataloader):
    stepny = 0
    tvarny = 0

    for i in range(len(dataloader.dataset)):
        if dataloader.dataset[i][1] == 0:
            stepny += 1
        else:
            tvarny += 1
    
    print(f"n of stepny: {stepny}")
    print(f"n of tvarny: {tvarny}")

data = LomyDataset()

#dataset length calculations
test_len = 200
length = data.loader.get_length()
train_len = length - test_len

train_dataset, test_dataset = random_split(data, [train_len, test_len]) #TODO

train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#inspect_classes(train_data)
#inspect_classes(test_data)

#for i in range(200):
#    inspect_dataset(i, test_data)

if __name__ == "__main__":
    #run neural networks
    net = ConvNeuralNet()
    net.run(train_data, test_data)