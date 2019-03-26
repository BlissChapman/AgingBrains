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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # 1x200x200
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(20,20), stride=2)
        # 10x91x91
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(11,11), stride=2)
        # 20x41x41
        self.fc21 = nn.Linear(20 * 41 * 41, 20)
        self.fc22 = nn.Linear(20 * 41 * 41, 20)
        self.fc3 = nn.Linear(20, 20 * 41 * 41)
        self.deconv1 = nn.ConvTranspose2d(20, 10, kernel_size=(11, 11), stride=2)
        self.deconv2 = nn.ConvTranspose2d(10, 1, kernel_size=(20, 20), stride=2)

    def encode(self, x):
        x = x.view(-1, 1, 200, 200)
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h2 = h2.view(-1, 20*41*41)
        return self.fc21(h2), self.fc22(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = h3.view(-1, 20, 41, 41)
        h4 = self.deconv1(h3)
        return torch.sigmoid(self.deconv2(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar