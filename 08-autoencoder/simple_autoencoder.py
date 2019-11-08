import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

img_folder = './mlp_img'
if not os.path.exists(img_folder):
    os.mkdir(img_folder)

def to_img(x):
    x = .5 * (x+1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 100
batch_size = 128
learning_rate =  1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    transforms.Normalize((.5,), (.5,))
])

dataset = MNIST('../data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = autoencoder()
use_gpu = torch.cuda.is_available()
if use_gpu: model = model.cuda()
summary(model, (1,28*28))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# start training
for epoch in range(1, num_epochs+1):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        if use_gpu: img = img.cuda()
        # forward propagation
        output = model(img)
        loss = criterion(output, img)
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'epoch [{epoch}/{num_epochs}, loss:{loss.item():.4f}]')
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, f'{img_folder}/image_{epoch}.png')
torch.save(model.state_dict(), './sim_autoencoder.pth')

# # ---- Test ----
# model = autoencoder()
# checkpoint = torch.load('./sim_autoencoder.pth')
# model.load_state_dict(checkpoint)

# # Test 1
# data = next(iter(dataloader))
# image = data[0][0][0].numpy()
# label = data[1][0].numpy()
# plt.imshow(image)
# plt.show()
# pred = model(Variable(torch.from_numpy(image.reshape(1,-1))))
# pred = pred.view(28,28).detach().numpy()
# plt.imshow(pred)
# plt.show()

# # Test 2
# from torchvision.datasets import FashionMNIST
# dataset =  FashionMNIST('../data', transform=img_transform, download=True)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# data = next(iter(dataloader))
# image = data[0][0][0].numpy()
# label = data[1][0].numpy()
# plt.imshow(image)
# plt.show()
# pred = model(Variable(torch.from_numpy(image.reshape(1,-1))))
# pred = pred.view(28,28).detach().numpy()
# plt.imshow(pred)
# plt.show()

# Conclusion : 
# model performs good on digital dataset - MNIST
# but not good on FashionMNIST 
# So this simple autoencoder seems most probably not good for generalization
