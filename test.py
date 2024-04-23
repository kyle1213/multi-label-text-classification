from utils import calc_accuracy
import torch


output = torch.tensor([[1, 1, 0], [0, 1, 0]]).float()
target = torch.tensor([[1, 0, 0], [0, 1, 0]])

print(calc_accuracy(output, target))