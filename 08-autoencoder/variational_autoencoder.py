import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

img_folder = './vae_img'
if not os.path.exists(img_folder):
    os.mkdir(img_folder)

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = MNIST('../data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

use_gpu = torch.cuda.is_available()

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 28*28)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(.5).exp_()
        if use_gpu:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
if use_gpu: model = model.cuda()
summary(model, (1, 28*28))

reconstruction_function = nn.MSELoss(reduction='sum')

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: original images
    mu: latent mean
    logvar: lantent log variance
    """
    BCE = reconstruction_function(recon_x, x) # MSE loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-.5)
    # KL divergence
    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, num_epochs+1):
    sum_loss = 0
    for batch_idx, (img, _) in enumerate(dataloader, 1):
        img = img.view(img.size(0), -1)
        img = Variable(img)
        if use_gpu: img = img.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss = loss_function(recon_batch, img, mu, logvar)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train epoch: {epoch}/{num_epochs} [{batch_idx*len(img)}/{len(dataloader.dataset)} \
                ({100.*batch_idx/len(dataloader):.0f}%)]\tLoss: {loss.item()/len(img):.6f}')
    print(f'=====> Epoch: {epoch}/{num_epochs} Average loss: {sum_loss/len(dataloader.dataset)}')
    if epoch % 10 == 0:
        pic = to_img(recon_batch.cpu().data)
        save_image(pic, f'{img_folder}/image_{epoch}.png')
torch.save(model.state_dict(), './vae.pth')

# # ---- Test ----
model = VAE()
checkpoint = torch.load('./vae.pth')
model.load_state_dict(checkpoint)

# # Test 1
data = next(iter(dataloader))
image = data[0][0][0].numpy()
label = data[1][0].numpy()
plt.imshow(image)
plt.show()
pred, _, _ = model(Variable(torch.from_numpy(image.reshape(1,-1))))
pred = to_img(pred.cpu().data)
pred = pred.view(28,28).detach().numpy()
plt.imshow(pred)
plt.show()

# Test 2
from torchvision.datasets import FashionMNIST
dataset =  FashionMNIST('../data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data = next(iter(dataloader))
image = data[0][0][0].numpy()
label = data[1][0].numpy()
plt.imshow(image)
plt.show()
pred, _, _ = model(Variable(torch.from_numpy(image.reshape(1,-1))))
pred = to_img(pred.cpu().data)
pred = pred.view(28,28).detach().numpy()
plt.imshow(pred)
plt.show()

# Conclusion : 
# model performs not so good on digital dataset - MNIST
# model performs bad on FashionMNIST 
# This approach seems not good for generalization either
