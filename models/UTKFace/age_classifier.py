import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.nn import functional as F

from processed_data import UTKFace
from utils import device


class AgeClassifier(nn.Module):
    
    def __init__(self):
        super(AgeClassifier, self).__init__()
        
        self.conv_layers = nn.Sequential(

            # (W−F+2P)/S+1
            # (200, 200) -> (97, 97)
            nn.Conv2d(1, 32, kernel_size=(8, 8), stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # (97, 97) -> (46, 46)
            nn.Conv2d(32, 64, kernel_size=(7, 7), stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # (46, 46) -> (22, 22)
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # (22, 22) -> (10, 10)
            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # (10, 10) -> 1
        self.fc1 = nn.Linear(10*10*256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        return x
    
    def _train_epoch_with_loader(self, train_loader):
        sum_train_loss = 0
        
        for batch_idx, (data_batch, label_batch) in enumerate(train_loader):
            torch.cuda.empty_cache() 
            
            data_batch = Variable(data_batch.to(device))
            label_batch = Variable(label_batch.to(device).float())
            
            self.zero_grad()
            label_predictions_batch = self(data_batch)
            loss = F.mse_loss(label_predictions_batch, label_batch)
            loss.backward()
            sum_train_loss += loss.item()
            self.optimizer.step()
            
        return sum_train_loss

    def train_model(self, num_epochs, model_output_path, log_interval=10):
        train_loader = torch.utils.data.DataLoader(
            UTKFace.Dataset(train=True),
            batch_size=64, shuffle=True)

        for epoch in range(1, num_epochs+1):
            self.train()
            sum_train_loss = self._train_epoch_with_loader(train_loader)
            print("EPOCH {0:10d} AVG AGE ESTIMATION ERROR: {1:.2f}".format(epoch, np.sqrt(sum_train_loss/len(train_loader))))
        
            if epoch % log_interval == 0:
                torch.save(self.state_dict(), output_model_name)
                self.eval()

