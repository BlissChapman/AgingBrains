import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
import torch.utils.data

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import device

class BaseModel(nn.Module):
    def __init__(self, model_name, data_name):
        """
        Constructor for a model that initializes parameters by reading the config file
        """
        super().__init__()
        
        self.model_name = model_name
        self.data_name = data_name
        self.df = pd.read_csv('processed_data/{}/df.csv'.format(self.data_name))
        
    
    def load(self):
        """
        Load model from version history
        :param version: index of the version to load, or None to load the most recent
        """
        raise Exception('Not Implemented')
    
    def test(self):
        """
        Tests the model and prints the results (losses)
        """
        raise Exception('Not Implemented')
    
    def train(self):
        """
        Train model one more epoch
        """
        raise Exception('Not Implemented')
    
    
    def predict(self, texts):
        """
        Returns the predicted values. Type is based on its feature
        """
        raise Exception('Not Implemented')
        
    @property
    def _model_path(self):
        return 'models/{}/'.format(self.model_name)
    @property
    def _weight_path(self):
        return self._model_path + '/weights/'
    @property
    def _stats_path(self):
        return self._model_path + '/stats/'
    
    def evaluate(self, epoch):
        """
        Evaluates the model from the testing data and saves the results into the stats path
        """
        raise Exception('Not Implemented')
            
    def save(self):
        """
        Save the model into the specified output path.
        This function is abstract and must be defined by the subclasses
        :param output_path: the path to save into
        """
        
        raise Exception('Not implemented')