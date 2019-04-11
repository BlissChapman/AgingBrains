from torchvision import datasets

class UTKFace(datasets.ImageFolder):
    def __init__(self, transform):
        super().__init__(root='data/UTKFace/processed/', transform=transform)
        