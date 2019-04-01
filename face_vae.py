import torch
import torch.utils.data
from torchvision.utils import save_image
from torch import optim
import numpy as np
from models.GreyUTKFace import VAE
import os
from processed_data import GreyUTKFace
import matplotlib.pyplot as plt

import argparse

from datetime import datetime

startTime = datetime.now()

# Parse args
parser = argparse.ArgumentParser(description='Face VAE')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# Set directory
model_dir = 'models/GreyUTKFace/VAE/'

# Load training data
train_dataset = GreyUTKFace.Dataset(train=True)
test_dataset = GreyUTKFace.Dataset(train=False)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128, shuffle=True)

# Startup the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE.Model()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(device)

test_losses = []
train_losses = []
weight_path = model_dir + 'weights/model.pt'

if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path))
    print('Loaded a saved model')
else:
    print('Starting model from scratch')
    
# Train
for epoch in range(1, args.epochs + 1):
    train_loss = VAE.train(model, train_loader, device, optimizer, epoch)
    test_loss = VAE.test(model, test_loader, device, optimizer, epoch)

    test_losses.append(test_loss)
    train_losses.append(train_loss)

    with torch.no_grad():
        sample = torch.randn(64, model.latent_space).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, VAE.input_w, VAE.input_h),
                   model_dir + 'results/sample_' + str(epoch) + '.png')

torch.save(model.state_dict(), weight_path)

if len(train_losses) > 1:
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.plot(np.arange(len(test_losses)), test_losses)
    plt.show()

print('\nTime elasped: ', datetime.now() - startTime)
