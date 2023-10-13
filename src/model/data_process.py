from torch.utils.data import Dataset, DataLoader, random_split

from models import BATCH_SIZE, ConvNeuralNet
from process import Loader
import os

class LomyDataset(Dataset):
    def __init__(self):
        print(os.getcwd())
        self.loader = Loader("dataset/lomy/stepnylom_jpg", "dataset/lomy/tvarnylom_jpg")
    
    def __getitem__(self, idx):
        data = self.loader.get(idx)
        return data[0], self.data[1] #X, y
    
    def __len__(self):
        return len(self.data)

data = LomyDataset()
train_dataset, test_dataset = random_split(data, [80, 10]) #TODO

train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

if __name__ == "__main__":
    #run neural networks
    net = ConvNeuralNet()
    net.run(train_data, test_data)