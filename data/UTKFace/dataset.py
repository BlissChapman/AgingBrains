from torchvision import datasets
import torch

class UTKFace(datasets.ImageFolder):
    def __init__(self, transform, label=None):
        root_dir = 'data/UTKFace/age/'
        target_transform = self.age_target_transform
        
        if label is not None:
            label = label.lower()
            if label == 'gender':
                root_dir = 'data/UTKFace/gender/'
                target_transform = None
            
        super().__init__(root=root_dir,
                         transform=transform,
                         target_transform=target_transform)
        
    def age_target_transform(self, idx):
        return int(self.classes[idx])