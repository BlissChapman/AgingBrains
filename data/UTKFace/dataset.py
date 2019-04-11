from torchvision import datasets

class UTKFace(datasets.ImageFolder):
    def __init__(self, transform):
        super().__init__(root='data/UTKFace/processed/', transform=transform, target_transform=self.target_transform)
    def target_transform(self, idx):
        return int(self.classes[idx])