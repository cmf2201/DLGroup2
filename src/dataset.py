import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.io import read_image
from PIL import Image
from pathlib import Path
import os
import glob
import pdb
import numpy as np

class DiffusionDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = transforms.Grayscale(num_output_channels=1)(torch.load(images_path).cpu())
        self.labels = torch.load(labels_path)

    def __len__(self):
        # Set the dataset size here
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label.item()
    
if __name__ == "__main__":
    images = "/home/ctnguyen/DLGroup2/src/tensor_images.pt"
    labels = "/home/ctnguyen/DLGroup2/src/labels.pt"
    dataset = DiffusionDataset(images, labels)
    dataloader = DataLoader(dataset)

    img, lbl = dataset[0]

    print(img)
    print(lbl)

