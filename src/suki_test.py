from torch import nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])  # Normalize for grayscale images

print("Start loading images")
images = torch.load("/home/skushwaha/DLGroup2/src/tensor_images.pt")
labels = torch.load("/home/skushwaha/DLGroup2/src/labels.pt")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataset = torch.utils.data.TensorDataset(images, labels)
combined_dataset = torch.utils.data.ConcatDataset([trainset, dataset])
print(images.shape)
print(labels)
print(dataset[0])
print(trainset[0])
print(combined_dataset[0])
print(combined_dataset[-1])

print("Done")

