import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils import data
import re

class Dataset(data.Dataset):
    """Face ages dataset."""

    def __init__(self, train, sample=False):
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
            print("WARNING: Sample of 512 files is being taken.")
            files = files[:512]
           
        split_idx = int(len(files) * 0.8)
        self.subset = files[:split_idx] if self.train else files[split_idx:]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        filename = self.subset[idx]
        age = self._parse_age(filename)
        img_name = os.path.join(self.root_dir, filename)
        img = self._read_data_from_disk(img_name)
        return img, np.array(age)
    
    @property
    def min_age(self):
        ages = list(map(lambda filename: self._parse_age(filename), self.subset))
        return min(ages)
    
    @property
    def max_age(self):
        ages = list(map(lambda filename: self._parse_age(filename), self.subset))
        return max(ages)
    
    def get_faces_of_age(self, target_age):
        """
        Returns all items with a given target age
        """
        ret = []
        for i, filename in enumerate(self.subset):
            age = self._parse_age(filename)
            if age == target_age:
                item = self[i]
                ret.append(item)
        return ret
        
    def _parse_age(self, filename):
        """
        The age feature is embedded in the filename. This function gets the age
        """
        match_obj = re.search('(\d+)_', filename)
        age = int(match_obj.group(1))
        return age
    
    def _read_data_from_disk(self, img_name):
        """
        Returns the face image of a given filename
        """
        # To PyTorch Tensor
        # numpy image: H x W
        # torch image: C X H X W
        image = io.imread(img_name)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float() / 255

        return image
