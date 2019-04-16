from torchvision import datasets
import torch

class CelebA(datasets.ImageFolder):
    
    def __init__(self, transform, label=None):
        root_dir = 'data/CelebA/young'

        if label is not None:
            label = label.lower()
            if label == 'male':
                root_dir = 'data/CelebA/male'

        super().__init__(root=root_dir,
                         transform=transform)