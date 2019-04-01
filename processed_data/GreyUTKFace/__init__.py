import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils import data
import re

class Dataset(data.Dataset):
    """Face ages dataset."""
    
    def _read_data_from_disk(self, img_name):
        image = io.imread(img_name)

        # To PyTorch Tensor
        # numpy image: H x W
        # torch image: C X H X W
        image = np.expand_dims(image, axis=0)

        image = torch.from_numpy(image).float() / 255

        match_obj = re.search('/(\d+)_', img_name)
        age = np.array(int(match_obj.group(1)))
        
        return image, age

    def __init__(self, train, sample=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = 'processed_data/GreyUTKFace/images/'
        self.train = train
        
        # Load all images into memory
        self.data = []
        files = os.listdir(self.root_dir)
        if sample:
            files = files[:1000]
        split_idx = int(len(files) * 0.8)
        self.subset = files[:split_idx] if self.train else files[split_idx:]
        
#         for filename in self.subset:
#             img_name = os.path.join(self.root_dir, filename)
#             data_item = self._read_data_from_disk(img_name)

#             self.data.append(data_item)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        filename = self.subset[idx]
        img_name = os.path.join(self.root_dir, filename)
        img, age = self._read_data_from_disk(img_name)
        return img, age
    

