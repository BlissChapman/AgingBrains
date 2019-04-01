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

input_w = 128
input_h = 128

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1, input_w, input_h), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(model, train_loader, device, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, avg_loss))
    return avg_loss


def test(model, test_loader, device, optimizer, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(128, 1, input_w, input_h)[:n]])
                save_image(comparison.cpu(), 
                           'models/GreyUTKFace/VAE/results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.latent_space = 50

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
    
