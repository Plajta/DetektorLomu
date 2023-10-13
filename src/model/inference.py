import torch
import model

model = torch.load("/src/model/saved/ConvNeuralNet9.pth")
model.eval()

