import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils import data
import re
import pandas as pd
from data import FaceDataset

class Dataset(FaceDataset):
    """CelebA"""

    def __init__(self, train, size, grey=False, sample=False):
        """
        """
        super().__init__(train, size, grey, sample)
        self.root_dir = 'data/celebA/'
        
        # Load all images into memory
        self.df = pd.read_csv(self.root_dir + 'list_attr_celeba.csv')
        self.partition_df = pd.read_csv(self.root_dir + 'list_eval_partition.csv')
        
        # Take only the rows that we're interested in
        self.partition_idx = 0 if self.train else 2
        self.df = self.df.iloc[self.partition_df[self.partition_df['partition'] == self.partition_idx].index]
        
        if self.sample:
            print("WARNING: Sample of 512 files is being taken.")
            self.df = self.df.iloc[:512]
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        series = self.df.iloc[idx]
        filename = series['image_id']
        young = series['Young']
        img_path = os.path.join(self.root_dir + 'img_align_celeba/', filename)
        img = self._resized_img_from_disk(img_path)
        return img, np.array(young)
    
