import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.nn import functional as F

from data import UTKFace
from tqdm import tqdm


class LinearRegression(nn.Module):
    
    def __init__(self):
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
    