import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os
from torch.utils.data import DataLoader
from torchsummary import summary

img_folder = './img'
if not os.path.exists(img_folder):
    os.mkdir(img_folder)

def to_img(x):
    x = .5 * (x+1)
    x = x.clamp(0, 1)
    x = x.view(-1, 1, 28, 28)
    return x

batch_size = 128
num_epoch = 100
z_dimention = 100
learning_rate = 3e-4

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5,), std=(.5,))
])

dataset = datasets.MNIST(
    root='../data', train=True, transform=img_transform, download=True
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.LeakyReLU(.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        return x

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x

D = discriminator()
G = generator()

use_gpu = torch.cuda.is_available()
if use_gpu: D, G = D.cuda(), G.cuda()

summary(D, (1, 28*28))
summary(G, (1, 100))

criterioin = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)

# start training
for epoch in range(1, num_epoch+1):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # ------- train discriminator -------
        img = img.view(num_img, -1)
        real_img = Variable(img)
        real_label = Variable(torch.ones(num_img).view(num_img, -1))
        fake_label = Variable(torch.zeros(num_img).view(num_img, -1))
        if use_gpu:
            real_img = real_img.cuda()
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()
        
        # compute loss of real_img
        real_out = D(real_img)
        d_loss_real = criterioin(real_out, real_label)
        real_scores = real_out # closer to 1 means better

        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimention))
        if use_gpu: z = z.cuda()
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterioin(fake_out, fake_label)
        fake_scores = fake_out #closer to 0 means better

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ------- train generator -------
        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimention))
        if use_gpu: z = z.cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterioin(output, real_label)

        # bp and optimize 
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch}/{num_epoch}], d_loss: {d_loss.item():.6f}, g_loss: {g_loss.item():.6f}, '
                f'D_real_scores: {real_scores.data.mean():.6f}, D_fake_scores: {fake_scores.data.mean():.6f}')
    if epoch == 1:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, f'{img_folder}/real_images.png')

    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, f'{img_folder}/fake_images-{epoch}.png')

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')





        
        
