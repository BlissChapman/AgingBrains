import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils import data
import re
import pandas as pd

class FaceDataset(data.Dataset):
    """CelebA"""

    def __init__(self, train, size, grey=False, sample=False):
        """
        """
        self.train = train
        self.grey = grey
        self.size = size
        self.sample = sample

    
    
    
    def _resized_img_from_disk(self, img_path):
        """
        Returns the face image of a given filename
        """
        # To PyTorch Tensor
        # numpy image: H x W x C
        # torch image: C X H X W
        image = io.imread(img_path, as_grey=self.grey)
        
        # Crop it to a square
        def crop_center(img,cropx,cropy):
            y,x = img.shape
            startx = x//2-(cropx//2)
            starty = y//2-(cropy//2)    
            return img[starty:starty+cropy,startx:startx+cropx]
       
        limiting_dim = min(image.shape[0], image.shape[1])
        image = crop_center(image, limiting_dim, limiting_dim)
        
        # Resize it
        image = transform.resize(image, (self.size, self.size))
        
        if self.grey:
            image = np.expand_dims(image, axis=0)
        else:
            image = image.transpose((2, 0, 1))
            
        # Scale it down to 0-1
        image = torch.from_numpy(image).float() / 255

        return image



