import torch
from torch import nn
from torch.autograd import Variable, grad
from torch.nn import functional as F

class Regression(nn.Module):
    
    def __init__(self):
        super().__init__()

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
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=(4,4), stride=2, padding=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.linears = nn.Sequential(
            nn.Linear(512 * 5 * 5, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.convs(x)
        out = self.linears(out.view(-1, 512 * 5 * 5))
        return out
    
    