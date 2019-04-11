import torch
from torch import nn
from torch.nn import functional as F

class SmallVAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.latent_space = 20

        # (W−F+2P)/S+1
        # 1x128x128
        self.encoder = nn.Sequential(
            nn.Linear(784, 400)
        )
        
        self.fc_mu = nn.Linear(400, self.latent_space)
        self.fc_logvar = nn.Linear(400, self.latent_space)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_space, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )
        
        
    def encode(self, x):
        out = x.view(-1, 784)
        out = self.encoder(out)
        return self.fc_mu(out), self.fc_logvar(out)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        out = self.decoder(z)
        return out.view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        
        # Reconstruction + KL divergence losses summed over all elements and batch
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
    