import torch
from torch import nn
from torch.autograd import Variable, grad
from torch.nn import functional as F

class Classifier(nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()

        # (W−F+2P)/S+1
        self.foo = nn.Sequential(
            nn.Linear(256 * 256, num_classes),
            nn.Softmax(1)
        )
        

    def forward(self, x):
        out = self.foo(x.view(-1, 256 * 256))
        return out
    
    