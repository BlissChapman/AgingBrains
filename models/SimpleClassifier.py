import torch
from torch import nn
from torch.autograd import Variable, grad
from torch.nn import functional as F

class SimpleClassifier(nn.Module):
    
    def __init__(self, input_dim, num_classes):
        super().__init__()

        # (W−F+2P)/S+1
        step = (input_dim-num_classes)//3
        self.linears = nn.Sequential(
            nn.Linear(input_dim, 2*step+num_classes),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(2*step+num_classes, 1*step+num_classes),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1*step+num_classes, num_classes),
            nn.Softmax(1)
        )
        

    def forward(self, x):
        out = self.linears(x)
        return out
    
    