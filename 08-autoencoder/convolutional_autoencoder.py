import torch
import torchvision
from torch import nn
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

img_folder = './dc_img'
if not os.path.exists(img_folder):
    os.mkdir(img_folder)

def to_img(x):
    x = .5 * (x+1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.5,], [.5])
])

dataset = MNIST('../data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1), # batch_size, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # batch_size, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1), # batch_size, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1) # batch_size, 8, 2, 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2), # batch_size, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1), # batch_size, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1), #batch_size, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

use_gpu = torch.cuda.is_available()
model = autoencoder()
if use_gpu: model = model.cuda()
summary(model, (1, 28, 28))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# start training
for epoch in range(1, num_epochs+1):
    for img, _ in dataloader:
        img = Variable(img)
        if use_gpu: img = img.cuda()
        # forward
        output = model(img)
        loss = criterion(output, img)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch {epoch}/{num_epochs}, loss:{loss.item():.4f}')
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, f'{img_folder}/image_{epoch}.png')
torch.save(model.state_dict(), './conv_autoencoder.pth')

# ---- Test ----
model = autoencoder()
checkpoint = torch.load('./conv_autoencoder.pth')
model.load_state_dict(checkpoint)
# Test 1
data = next(iter(dataloader))
image = data[0][0][0].numpy()
label = data[1][0].numpy()
plt.imshow(image)
plt.show()
pred = model(Variable(torch.from_numpy(image.reshape(-1,1,28,28))))
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
pred = model(Variable(torch.from_numpy(image.reshape(-1,1,28,28))))
pred = pred.view(28,28).detach().numpy()
plt.imshow(pred)
plt.show()

# Conclusion : 
# model performs not well on digital dataset - MNIST
# and bad on FashionMNIST either.
# but a little bit better than the simple linear autoencoder 

