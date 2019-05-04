from torchvision import datasets
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image

class CelebA(Dataset):
    
    def __init__(self, transform):
        super().__init__()
        self.root_dir = 'data/CelebA/'
        self.df = pd.read_csv(self.root_dir + 'list_attr_celeba.csv')
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.root_dir + 'raw/{}'.format(row['image_id']))
        img = self.transform(img)
        label = np.array(list(map(lambda x: 0 if x == -1 else 1, row.values[1:])))
        
        return img, label