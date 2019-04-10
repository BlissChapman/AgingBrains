import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils import data
import re
from data import FaceDataset

class Dataset(FaceDataset):
    """Face ages dataset."""

    def __init__(self, train, size, grey=False, sample=False):
        """
        """
        super().__init__(train, size, grey, sample)
        self.root_dir = 'data/UTKFace/images/'
        
        # Load all images into memory
        self.data = []
        files = os.listdir(self.root_dir)
        
        if self.sample:
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
        img = self._resized_img_from_disk(img_name)
        
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

