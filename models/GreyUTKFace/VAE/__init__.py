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
from pdb import set_trace

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.input_w = 128
        self.input_h = 128
        self.latent_space = 50

        # (W−F+2P)/S+1
        
        # 1x128x128
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(5,5), stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5,5), stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(5,5), stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(5,5), stride=2, padding=2)
        
        self.fc_mu = nn.Linear(512 * 8 * 8, self.latent_space)
        self.fc_logvar = nn.Linear(512 * 8 * 8, self.latent_space)
        self.fc_reshape = nn.Linear(self.latent_space, 1024 * 8 * 8)
        
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=(5,5), stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(5,5), stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(5,5), stride=2, padding=2, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=(5,5), stride=2, padding=2, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=(5,5), stride=1, padding=2)
        self.deconv6 = nn.ConvTranspose2d(32, 1, kernel_size=1)

    def encode(self, x):
        x = x.view(-1, 1, 128, 128)
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h4 = h4.view(-1, 512 * 8 * 8)
        return self.fc_mu(h4), self.fc_logvar(h4)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h6 = F.relu(self.fc_reshape(z))
        h6 = h6.view(-1, 1024, 8, 8)
        h7 = F.relu(self.deconv1(h6))
        h8 = F.relu(self.deconv2(h7))
        h9 = F.relu(self.deconv3(h8))
        h10 = F.relu(self.deconv4(h9))
        h11 = F.relu(self.deconv5(h10))
        h12 = self.deconv6(h11)
        return torch.sigmoid(h12)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar