from __future__ import print_function
import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import shutil
import torch
import torch.utils.data

from pdb import set_trace
from processed_data import GreyUTKFace
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import device


class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        
        self.input_w = 128
        self.input_h = 128
        self.latent_space = 50

        self.dir = 'models/GreyUTKFace/VAE/'
        
        self.epochs_trained = 0

        # (W−F+2P)/S+1
        # 1x128x128
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5,5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(5,5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(5,5), stride=2, padding=2),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256 * 8 * 8, self.latent_space)
        self.fc_logvar = nn.Linear(256 * 8 * 8, self.latent_space)
        self.fc_reshape = nn.Linear(self.latent_space, 256 * 8 * 8)
        
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(5,5), stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(5,5), stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(5,5), stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(5,5), stride=2, padding=2, output_padding=1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
    def load(self):
        self.load_state_dict(torch.load(self.dir + 'weights/model.pt'))
        with open(self.dir + 'weights/epochs_trained.txt', 'r') as f:
            self.epochs_trained = int(f.read())
            
    def save(self):
        torch.save(self.state_dict(), self.dir + 'weights/model.pt')
        with open(self.dir + 'weights/epochs_trained.txt', 'w') as f:
            f.write(str(self.epochs_trained))
        
    def encode(self, x):
        out = x.view(-1, 1, 128, 128)
        out = self.convs(out)
        out = out.view(-1, 256 * 8 * 8)
        return self.fc_mu(out), self.fc_logvar(out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        out = F.relu(self.fc_reshape(z))
        out = out.view(-1, 256, 8, 8)
        out = self.deconvs(out)
        return torch.sigmoid(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        
        # Reconstruction + KL divergence losses summed over all elements and batch
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1, self.input_w, self.input_h), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def _train_epoch_with_loader(self, train_loader):
        self.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self(data)
            loss = self.loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()            
        avg_loss = train_loss / len(train_loader.dataset)
        return avg_loss
    
    def train_model(self, num_epochs, sample=False, log_interval=10):
        
        print("Loading dataset...")
        train_loader = torch.utils.data.DataLoader(
            GreyUTKFace.Dataset(train=True, sample=sample),
            batch_size=128, shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(
            GreyUTKFace.Dataset(train=False, sample=sample),
            batch_size=128, shuffle=True)
        
        print("Training...")
        for epoch in range(1, num_epochs+1):
            self.train()
            avg_train_loss = self._train_epoch_with_loader(train_loader)
            print("EPOCH {0:10d} AVG LOSS: {1}".format(epoch, avg_train_loss))
        
            if epoch % log_interval == 0:
                
                # Plot reconstructions
                for i, (data, _) in enumerate(test_loader):
                    data = data.to(device)
                    recon_batch, mu, logvar = self(data)
                    if i == 0:
                        n = min(data.size(0), 8)
                        comparison = torch.cat([data[:n],
                                              recon_batch.view(128, 1, self.input_w, self.input_h)[:n]])
                        save_image(comparison.cpu(), 
                                   'models/GreyUTKFace/VAE/results/reconstruction_' + str(epoch) + '.png', nrow=n)
                        break
                    
                # Save model weights
                self.save()