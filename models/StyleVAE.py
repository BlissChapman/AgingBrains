import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import models

import copy

device = torch.device("cuda")
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleRep(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        G = gram_matrix(input)
        self.rep = G
        return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
        

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                              style_layers=['conv_1','conv_2','conv_3']):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    style_reps = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
            # Replace with Average Pool
            layer = nn.AvgPool2d(2)

        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers:
            # add style loss:
            style_rep = StyleRep()
            model.add_module("style_rep_{}".format(i), style_rep)
            style_reps.append(style_rep)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], StyleRep):
            break

    model = model[:(i + 1)]

    return model, style_reps

style_model, style_reps = get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std)



class StyleVAE(nn.Module):
    
    def __init__(self, num_channels=3):
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
            nn.Conv2d(128, 256, kernel_size=(5,5), stride=2, padding=2, bias=True),
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
            nn.ConvTranspose2d(16, num_channels, kernel_size=(5,5), stride=2, padding=2, output_padding=1, bias=True),
            nn.Sigmoid()
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
    def encode(self, x):
        out = self.convs(x)
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
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss(self, recon_x, x, mu, logvar):
        
        total_style_loss = torch.tensor(0, dtype=torch.float, device=device)
        
        true_reps = []
        style_model(x)
        for sp in style_reps:
            true_reps.append(sp.rep.clone())
        
        reconst_reps = []
        style_model(recon_x)
        for sp in style_reps:
            reconst_reps.append(sp.rep.clone())
        
        
        for true_rep, reconst_rep in zip(true_reps, reconst_reps):
            total_style_loss += F.mse_loss(reconst_rep, true_rep)
        
        total_style_loss *= 1000000*1000
        
        # Reconstruction + KL divergence losses summed over all elements and batch
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.num_channels, self.input_size, self.input_size), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        BETA = 0.001
        
#         print("{:.4f} {:.4f} {:.4f}".format(BCE.item(), total_style_loss.item(), KLD.item()))
        
        return (BCE + total_style_loss) + BETA*KLD
    