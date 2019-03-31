import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils import data
import re

class Dataset(data.Dataset):
    """Face ages dataset."""

    def __init__(self, train):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = 'processed_data/UTKFace/images/'
        self.train = train
        
        # Load all images into memory
        self.data = []
        files = os.listdir(self.root_dir)
        split_idx = int(len(files) * 0.8)
        subset = files[:split_idx] if self.train else files[split_idx:]
        for filename in subset:
            img_name = os.path.join(self.root_dir, filename)
            image = io.imread(img_name)[:,:, 0]
            
            # To PyTorch Tensor
            # numpy image: H x W
            # torch image: C X H X W
            image = np.expand_dims(image, axis=0)
            
            image = torch.from_numpy(image).float() / 255

            match_obj = re.search('/(\d+)_', img_name)
            age = np.array(int(match_obj.group(1)))

            self.data.append((image, age))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

