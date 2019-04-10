import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.nn import functional as F

from models import BaseModel
from processed_data import GreyUTKFace
from tqdm import tqdm
from utils import device


class Model(BaseModel):
    
    def __init__(self, device):
        super().__init__('GreyUTKFaceAgeClassifier', GreyUTKFace.Dataset, device)
                
        self._batch_size = 32
        self.conv_layers = nn.Sequential(

            # (W−F+2P)/S+1
            # (128, 128) -> (63, 63)
            nn.Conv2d(1, 32, kernel_size=(4, 4), stride=2),
            nn.ReLU(),

            # (63, 63) -> (31, 31)
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2),
            nn.ReLU(),

            # (31, 31) -> (15, 15)
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2),
            nn.ReLU(),

            # (15, 15) -> (7, 7)
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2),
            nn.ReLU(),
        )

        # (7, 7) -> 1
        self.fc1 = nn.Linear(7*7*256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

#         self.fc1 = nn.Linear(128*128, 64*128)
#         self.fc2 = nn.Linear(64*128, 1)
#         self.optimizer = optim.SGD(self.parameters(), lr=1e-5)


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)

#         x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
        return x
    
    def train_an_epoch(self, sample=False):
        super().train_an_epoch()
        
        train_loader = self.train_loader(sample)
        
        self.train()
        train_loss = 0
        for data_batch, label_batch in tqdm(train_loader):
            torch.cuda.empty_cache() 
            
            data_batch = Variable(data_batch.to(device))
            label_batch = Variable(label_batch.to(device).float())
            
            self.optimizer.zero_grad()
            label_predictions_batch = self(data_batch)
            loss = F.mse_loss(label_predictions_batch, label_batch)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
           
        train_loss /= len(train_loader.dataset)
        return train_loss

    def test(self, sample=False):
        super().test()

        test_loader = self.test_loader(sample)

        test_loss = 0
        with torch.no_grad():
            for data_batch, label_batch in tqdm(test_loader):
                torch.cuda.empty_cache() 

                data_batch = Variable(data_batch.to(device))
                label_batch = Variable(label_batch.to(device).float())

                label_predictions_batch = self(data_batch)
                test_loss += F.mse_loss(label_predictions_batch, label_batch).item()

        test_loss /= len(test_loader.dataset)
        return test_loss
        
    def evaluate(self):
        super().evaluate()
        
        self.eval()
        test_loader = self.test_loader(False)

        # Print mean and variance of true/predicted ages
        true_ages = []
        predicted_ages = []
        with torch.no_grad():
            for data_batch, label_batch in tqdm(test_loader):
                torch.cuda.empty_cache() 

                data_batch = Variable(data_batch.to(device))
                label_batch = Variable(label_batch.to(device).float())
                label_predictions_batch = self(data_batch)
                
                true_ages.append(label_batch.cpu().data)
                predicted_ages.append(label_predictions_batch.cpu().data)

        predicted_ages = np.concatenate(predicted_ages, axis=0)
        true_ages = np.concatenate(true_ages, axis=0)
        
        print("MEAN TRUE AGES: {0:.2f}, VAR TRUE AGES: {1:.2f}".format(np.mean(true_ages), np.var(true_ages)))
        print("MEAN PREDICTED AGES: {0:.2f}, VAR PREDICTED AGES: {1:.2f}".format(np.mean(predicted_ages), np.var(predicted_ages)))