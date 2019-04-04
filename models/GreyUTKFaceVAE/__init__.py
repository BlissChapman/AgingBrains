import numpy as np
import shutil
import torch
import torch.utils.data

from processed_data import GreyUTKFace
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import device

from models import BaseModel


class Model(BaseModel):
    
    def __init__(self, device):
        super().__init__('GreyUTKFaceVAE', GreyUTKFace.Dataset, device)
        
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
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
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
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1, self.Dataset.width, self.Dataset.height), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
    
    def train_an_epoch(self, sample=False):
        super().train_an_epoch()
        
        train_loader = self.train_loader(sample)
        
        self.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self(data)
            loss = self.loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()            
        avg_loss = train_loss / len(train_loader.dataset)
        return avg_loss
    
    def test(self, sample=False):
        super().test()
        
        test_loader = self.test_loader(sample)
        
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self(data)
                test_loss += self.loss(recon_batch, data, mu, logvar).item()

        test_loss /= len(test_loader.dataset)
        return test_loss
        
    def evaluate(self, epoch):
        super().evaluate(epoch)
        
        self.eval()
        test_loader = self.test_loader(sample=False)
        
        # Reconstructions
        for data, _ in test_loader:
            data = data.to(self.device)
            recon_batch, mu, logvar = self(data)
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                  recon_batch.view(self._batch_size, 1, self.Dataset.width, self.Dataset.height)[:n]])
            save_image(comparison.cpu(), 
                       self._results_path + 'reconstruction_' + str(epoch) + '.png', nrow=n)
            break
        
        # Generate new data
        with torch.no_grad():
            sample = torch.randn(64, self.latent_space).to(device)
            sample = self.decode(sample).cpu()
            save_image(sample.view(64, 1, self.Dataset.width, self.Dataset.height),
                       self._results_path + 'sample_' + str(epoch) + '.png')