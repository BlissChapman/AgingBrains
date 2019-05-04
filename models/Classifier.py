import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.nn import functional as F
from torch import optim
from torchvision.utils import save_image

class Classifier(nn.Module):

    def __init__(self, input_dimensionality, output_dimensionality):
        super(Classifier, self).__init__()

        self.input_dimensionality = input_dimensionality
        self.output_dimensionality = output_dimensionality
        
        layer_sizes = [
            int(input_dimensionality),
            int(2*input_dimensionality),
            int(2*input_dimensionality),
            int(input_dimensionality - 0.33*(input_dimensionality-output_dimensionality)),
            int(input_dimensionality - 0.67*(input_dimensionality-output_dimensionality)),
            int(output_dimensionality),
        ]

        self.fc1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.fc4 = nn.Linear(layer_sizes[3], layer_sizes[4])
        self.fc5 = nn.Linear(layer_sizes[4], layer_sizes[5])        
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3)
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3)
        
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.3)
        
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.3)
        
        x = torch.sigmoid(self.fc5(x))
        
        return x