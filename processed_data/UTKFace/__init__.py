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
        self.filenames = os.listdir(self.root_dir)
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.filenames[idx])
        image = io.imread(img_name, as_grey=True).reshape(200, 200, 1) / 255
        
        if self.transform:
            image = self.transform(image).float()
        else:
            image = torch.Tensor(image).float()
        
        match_obj = re.search('/(\d+)_', img_name)
        age = np.array(int(match_obj.group(1)))

        return image, age