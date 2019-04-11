import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.input_size = 256
        self.latent_space = 50

        # (W−F+2P)/S+1
        self.convs = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5,5), stride=2, padding=2, bias=False),
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
        
        self.fc_mu = nn.Linear(256 * 8 * 8, self.latent_space)
        self.fc_logvar = nn.Linear(256 * 8 * 8, self.latent_space)
        self.fc_reshape = nn.Linear(self.latent_space, 256 * 8 * 8)
        
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
            nn.ConvTranspose2d(16, 1, kernel_size=(5,5), stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(1),
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
    
    def loss(self, recon_x, x, mu, logvar):
        
        # Reconstruction + KL divergence losses summed over all elements and batch
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1, self.input_size, self.input_size), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD
    