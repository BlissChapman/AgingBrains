from torchvision import datasets
import torch

class CelebA(datasets.ImageFolder):
    def __init__(self, transform):
        root_dir = 'data/CelebA/'
        super().__init__(root=root_dir, transform=transform)