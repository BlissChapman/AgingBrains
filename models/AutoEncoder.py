import torch
from torch import nn
from torch.nn import functional as F

class AutoEncoder(nn.Module):
    
    def __init__(self, num_channels=1):
        super().__init__()
        
        self.input_size = 256
        self.latent_space = 50
        self.num_channels = num_channels

        # (W−F+2P)/S+1
        self.convs = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=(5,5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(5,5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(5,5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(5,5), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.to_latent = nn.Linear(256 * 8 * 8, self.latent_space)
        
        self.from_latent = nn.Sequential(
            nn.Linear(self.latent_space, 256 * 8 * 8),
            nn.ReLU()
        )
        
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(5,5), stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(5,5), stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(5,5), stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(5,5), stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, num_channels, kernel_size=(5,5), stride=2, padding=2, output_padding=1, bias=False),
            nn.Sigmoid()
        )
        
        
    def encode(self, x):
        out = self.convs(x)
        out = out.view(-1, 256 * 8 * 8)
        out = self.to_latent(out)
        return out

    def decode(self, z):
        out = self.from_latent(z)
        out = out.view(-1, 256, 8, 8)
        out = self.deconvs(out)
        return out

    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)
    
    def loss(self, recon_x, x):
        
        # Reconstruction + KL divergence losses summed over all elements and batch
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.num_channels, self.input_size, self.input_size), reduction='sum')

        return BCE
    