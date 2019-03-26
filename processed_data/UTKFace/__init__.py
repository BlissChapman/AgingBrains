import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils import data
import re

class Dataset(data.Dataset):
    """Face ages dataset."""

    def __init__(self, train, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = 'processed_data/UTKFace/images/'
        self.train = train
        self.transform = transform

        self.data = []
        for filename in os.listdir(self.root_dir):
            img_name = os.path.join(self.root_dir, filename)
            image = io.imread(img_name, as_grey=True).reshape(1, 200, 200)
            
            if self.transform:
                image = self.transform(image).float()
            else:
                image = torch.Tensor(image).float()

            match_obj = re.search('/(\d+)_', img_name)
            age = np.array(int(match_obj.group(1)))

            self.data.append((image, age))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return data[idx]
