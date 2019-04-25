import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.nn import functional as F
from torch import optim
from torchvision.utils import save_image

class LatentDiscriminator(nn.Module):

    def __init__(self, latent_dimensionality, num_attributes):
        super(LatentDiscriminator, self).__init__()

        self.num_attributes = num_attributes
        layer_sizes = [
            int(latent_dimensionality),
            int(latent_dimensionality - 0.2*(latent_dimensionality-num_attributes)),
            int(latent_dimensionality - 0.4*(latent_dimensionality-num_attributes)),
            int(latent_dimensionality - 0.6*(latent_dimensionality-num_attributes)),
            int(latent_dimensionality - 0.8*(latent_dimensionality-num_attributes)),
            int(num_attributes),
        ]

        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.fc4 = nn.Linear(layer_sizes[3], layer_sizes[4])
        self.fc5 = nn.Linear(layer_sizes[4], layer_sizes[5])
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        F.dropout(x, p=0.3)
        
        x = F.relu(self.fc2(x))
        F.dropout(x, p=0.3)
        
        x = F.relu(self.fc3(x))
        F.dropout(x, p=0.3)
        
        x = F.relu(self.fc4(x))
        F.dropout(x, p=0.3)
        
        x = F.sigmoid(self.fc5(x))
        
        return x