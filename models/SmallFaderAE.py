import torch
from torch import nn
from torch.nn import functional as F

class SmallFaderAE(nn.Module):
    
    def __init__(self, num_attributes):
        super().__init__()
        
        self.latent_space = 20
        self.num_attributes = num_attributes

        # (W−F+2P)/S+1
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, self.latent_space)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_space + num_attributes, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )
        
        
    def encode(self, x):
        out = x.view(-1, 784)
        out = self.encoder(out)
        return out

    def decode(self, z, attributes):
        attributes = attributes.view(-1, self.num_attributes).float()
        
        h = torch.cat((z, attributes), 1)
        
        out = self.decoder(h)
        return out.view(-1, 1, 28, 28)
    