import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
import os
from torchsummary import summary

img_folder = './dc_img'
if not os.path.exists(img_folder):
    os.mkdir(img_folder)

def to_img(x):
    x = .5 * (x+1)
    x = x.clamp(0, 1)
    x = x.view(-1, 1, 28, 28)
    return x

batch_size = 128
num_epoch = 100
z_dimension = 100 # noise dimension
learning_rate = 3e-4

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5,), (.5,))
])

dataset = datasets.MNIST('../data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # num_workers=4

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), # b, 32, 28, 28
            nn.LeakyReLU(.2, True),
            nn.AvgPool2d(2, stride=2) # b, 32, 12, 12
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2), # b, 64, 14, 14
            nn.LeakyReLU(.2, True),
            nn.AvgPool2d(2, stride=2) # batch, 64, 7, 7
        )

        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.LeakyReLU(.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature) # b, 3136=1*56*56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1), # b, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1), # b, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2), # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x

D, G = discriminator(), generator(z_dimension, 3136)
use_gpu = torch.cuda.is_available()
if use_gpu: D, G = D.cuda(), G.cuda()
summary(D, (1, 28, 28))
summary(G, (1, 100))
criterion = nn.BCELoss() # binary cross netropy loss
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)

# start training
for epoch in range(1, num_epoch+1):
    for i, (img, _) in enumerate(dataloader):
        num_img = img.size(0)
        # ------- train discriminator -------
        real_img = Variable(img)
        real_label = Variable(torch.ones(num_img).reshape(num_img, -1))
        fake_label = Variable(torch.zeros(num_img).reshape(num_img, -1))
        if use_gpu:
            real_img = real_img.cuda()
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()

        # compute loss of real images
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out # closer to 1 means better

        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension))
        if use_gpu: z = z.cuda()
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out # closer to 0 means better

        # bp and optimization
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ------- train generator ------
        # compute loss of fake images
        z = Variable(torch.randn(num_img, z_dimension))
        if use_gpu: z = z.cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)

        # bp and optimization
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
torch.save(G.state_dict(), './conv_generator.pth')
torch.save(D.state_dict(), './conv_discriminator.pth')

