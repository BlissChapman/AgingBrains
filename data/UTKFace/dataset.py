from torchvision import datasets
import torch

class UTKFace(datasets.ImageFolder):
    def __init__(self, transform, label=None):
        root_dir = 'data/UTKFace/age/'
        target_transform = self.age_target_transform
        
        label = label.lower()
        if label == 'gender':
            root_dir = 'data/UTKFace/gender/'
            target_transform = self.gender_target_transform
            
        super().__init__(root=root_dir, transform=transform, target_transform=target_transform)
        
    def age_target_transform(self, idx):
        return int(self.classes[idx])
    
    def gender_target_transform(self, idx):
        foo = torch.zeros(2, dtype=torch.float)
        foo[idx] = 1
        return foo