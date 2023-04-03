from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.input_size = input_size
        self.class_size = class_size
        self.latent_size = latent_size
        self.units = 400
        self.encode1 = nn.Linear(input_size + self.class_size, self.units)
        self.encode2 = nn.Linear(self.units, self.units//2)
        self.encode3 = nn.Linear(self.units//2, latent_size)
        self.encode4 = nn.Linear(self.units//2, latent_size)
        self.decode1 = nn.Linear(latent_size + self.class_size, self.units//2)
        self.decode2 = nn.Linear(self.units//2, self.units)
        self.decode3 = nn.Linear(self.units, self.input_size)



    def encoding_model(self, x, c):
        theinput = torch.cat((x.float(), c.float()), 1)
        output = self.encode1(theinput)
        output = self.encode2(output)
        mu = self.encode3(output)
        logvar = self.encode4(output)
        return mu, logvar

    def decoding_model(self, z, c):
        z_input = torch.cat((z.float(), c.float()), 1)
        output = self.decode1(z_input)
        output = self.decode2(output)
        x_hat = self.decode3(output)
        return x_hat

    def forward(self, x, c):
        mu, logvar = self.encoding_model(x, c)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoding_model(z, c)
        return x_hat, mu, logvar
   
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        epsilon = Variable(std.data.new(std.size()).normal_())
        return epsilon.mul(std) + mu


def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return Variable(targets)

def loss_function(x_hat, x, mu, logvar):
    reconstruction_function = nn.BCEWithLogitsLoss()
    reconstruction_function.size_average = True
    reconstruction_loss = reconstruction_function(x_hat, x)
    kl_divergence = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, dim=1)
    kl_divergence = torch.sum(kl_divergence, dim=0)
    loss = (reconstruction_loss + kl_divergence)
    return loss

def train(epoch, model, train_loader, optimizer, num_classes):
    model.train()
    train_loss = 0
    for batch_number, (data, labels) in enumerate(train_loader):
        data = Variable(data).view(data.shape[0], -1)
        labels = one_hot(labels, num_classes)
        reconstruction_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(reconstruction_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_number % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_number * len(data), len(train_loader.dataset),
                100. * batch_number / len(train_loader),
                loss.data / len(data)))
kwargs = {}
input_size = 28 * 28
units = 400
batch_size = 32
latent_size = 20
num_classes = 10
num_epochs = 11

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, **kwargs)
test_loader=torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=False,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=False, **kwargs)
model = CVAE(input_size, latent_size, num_classes)
def test():
    model.eval()
    test_loss= 0
    with torch.no_grad():
        for batch_number, (data, labels) in enumerate(test_loader):
            data = Variable(data).view(data.shape[0], -1)
            data, labels = data, one_hot(labels, num_classes)
            reconstruction_batch, mu, logvar = model(data, labels)
            # sum up batch loss
            test_loss += loss_function(reconstruction_batch, data, mu, logvar).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    
def main():
    input_size = 28 * 28
    units = 400
    batch_size = 32
    latent_size = 20
    num_classes = 10
    num_epochs = 11
   
    

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1, num_epochs):
        train(epoch, model, train_loader, optimizer, num_classes)
        test()

    c = torch.eye(num_classes, num_classes)
    c = Variable(c)
    z = Variable((torch.randn(num_classes, latent_size)))
    samples = model.decoding_model(z, c).data.cpu().numpy()

    fig = plt.figure(figsize=(10, 1))
    gridspec2 = gridspec.GridSpec(1, 10)
    gridspec2.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gridspec2[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    plt.show()


if __name__ == "__main__":
    main()
