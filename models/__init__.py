import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
import torch.utils.data

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import os

class BaseModel(nn.Module):
    def __init__(self, model_name, Dataset, input_size, device):
        """
        Constructor for a model
        """
        super().__init__()
        
        self.model_name = model_name
        self.Dataset = Dataset
        self.device = device
        self.epochs_trained = 0
        self._batch_size = 128
        self.input_size = input_size

        
    def save(self):
        """
        Save the model into the specified output path.
        """
        weight_path = self._model_path + 'weights/'
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
            
        torch.save(self.state_dict(), weight_path + 'model.pt')
        with open(self._model_path + 'weights/epochs_trained.txt', 'w') as f:
            f.write(str(self.epochs_trained))
            
    def load(self):
        """
        Load model
        """
        try:
            self.load_state_dict(torch.load(self._model_path + 'weights/model.pt'))
        except Exception as e:
            print(e)
            
        try:
            with open(self._model_path + 'weights/epochs_trained.txt', 'r') as f:
                self.epochs_trained = int(f.read())
        except:
            raise Exception('Epochs trained missing')
            
    def save_exists(self):
        return os.path.exists(self._model_path + 'weights/model.pt')

    def test(self):
        """
        Tests the model on the test set and prints the results (losses)
        """
        self.to(self.device)
    
    def train_an_epoch(self):
        """
        Train model one more epoch
        """
        self.epochs_trained += 1
        self.to(self.device)
        
    def evaluate(self):
        """
        Evaluates the model from the testing data and saves the results into the results path
        """
        results_path = self._model_path + 'results/'
        if not os.path.exists(results_path):
            os.makedirs(results_path)
    
    def train_loader(self, sample):
        return torch.utils.data.DataLoader(
            self.Dataset(train=True, sample=sample, size=self.input_size, grey=True),
            batch_size=self._batch_size, shuffle=True, num_workers=4)
    
    def test_loader(self, sample):
        return torch.utils.data.DataLoader(
            self.Dataset(train=False, sample=sample, size=self.input_size, grey=True),
            batch_size=self._batch_size, shuffle=True, num_workers=4)
        
    @property
    def _model_path(self):
        return 'models/{}/'.format(self.model_name)
    @property
    def _weight_path(self):
        return self._model_path + '/weights/'
    @property
    def _results_path(self):
        return self._model_path + '/results/'
    
    
            