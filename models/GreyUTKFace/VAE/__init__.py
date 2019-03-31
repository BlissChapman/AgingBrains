from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import copy
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # (W−F+2P)/S+1
        
        # 1x200x200
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(10,10), stride=2)
        # (200 - 10) / 2 + 1 = 95 + 1 = 96
        # 8x96x96
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(8,8), stride=2)
        # (96 - 8) / 2 + 1 = 44 + 1 = 45
        # 16x45x45
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(5,5), stride=2)
        # (45 - 5) / 2 + 1 = 20 + 1 = 21
        # 32x21x21
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=1)
        # (21 - 3) / 1 + 1 = 18 + 1 = 19
        # 64x19x19
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=2)
        # (19 - 3) / 2 + 1 = 8 + 1 = 9
        # 128 x 9 x 9
        self.mu = nn.Linear(128 * 9 * 9, 32)
        self.logvar = nn.Linear(128 * 9 * 9, 32)
        self.fc1 = nn.Linear(32, 128 * 9 * 9)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=(3,3), stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=(5,5), stride=2)
        self.deconv4 = nn.ConvTranspose2d(16, 8, kernel_size=(8,8), stride=2)
        self.deconv5 = nn.ConvTranspose2d(8, 1, kernel_size=(10,10), stride=2)

    def encode(self, x):
        x = x.view(-1, 1, 200, 200)
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = F.relu(self.conv5(h4))
        h5 = h5.view(-1, 128 * 9 * 9)
        return self.mu(h5), self.logvar(h5)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h6 = F.relu(self.fc1(z))
        h6 = h6.view(-1, 128, 9, 9)
        h7 = self.deconv1(h6)
        h8 = self.deconv2(h7)
        h9 = self.deconv3(h8)
        h10 = self.deconv4(h9)
        h11 = self.deconv5(h10)
        return torch.sigmoid(h11)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar