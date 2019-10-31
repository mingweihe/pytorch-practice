import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import numpy as np
from torchsummary import summary
# import matplotlib.pyplot as plt
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# import matplotlib
# matplotlib.use('Agg')

# hyper paramerters
batch_size = 64
learning_rate = 1e-3
num_epochs = 100

# download Fashion MNIST dataset, show the first image
# train_set = datasets.FashionMNIST(
#     root='../data', train=True, download=True)
# plt.imshow(np.array(train_set[0][0]), cmap='gray')
# plt.show()

# load train & test data, transform to tensor, and normalize them
train_set = datasets.FashionMNIST(
    root='../data', train=True, transform=transforms.ToTensor(), download=True)
test_set = datasets.FashionMNIST(
    root='../data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

num_train = len(train_set)
num_test = len(test_set)

print(f'length of train set: {num_train}, shape of each data: {train_set[0][0].shape}')
print(f'length of train loader: {len(train_loader)}')

print(f'length of train set: {num_test}')
print(f'length of test loader: {len(test_loader)}')

# define the logistic regression model
class logistic_regression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(logistic_regression, self).__init__()
        self.logistic = nn.Linear(in_dim, n_class)
        
    def forward(self, x):
        out = self.logistic(x)
        return out

model = logistic_regression(28 * 28, 10)
print(summary(model, (1, 28 * 28)))

use_gpu = torch.cuda.is_available()
if use_gpu: model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# start training
for epoch in range(1, num_epochs+1):
    print('*' * 20)
    print(f'epoch {epoch}')
    since = time.time()
    sum_loss = .0
    sum_acc = .0
    model.train()
    # ********* training *************
    for i, (img, label) in enumerate(train_loader, 1):
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        # forward propagation
        out = model(img)
        loss = criterion(out, label)
        sum_loss += loss.item()
        _, pred = torch.max(out, 1) # 1 -> compared dimension
        sum_acc += (pred==label).float().mean()
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 300 == 0:
            print(f'[Epoch: {epoch}/{num_epochs}, step: {i}] Loss: {sum_loss/i: .6f}, Accuracy: {sum_acc/i: .6f}')
        
    print(f'Finish {epoch} epoch, Loss: {sum_loss/i: .6f}, Accuracy: {sum_acc/i: .6f}')
    model.eval()
    sum_loss = .0
    sum_acc = .0
    # ********* evaluation *************
    for img, label in test_loader:
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        with torch.no_grad():
            out = model(img)
            loss = criterion(out, label)
        sum_loss += loss.item()
        _, pred = torch.max(out, 1)
        sum_acc += (pred==label).float().mean()
    print(f'Test loss: {sum_loss/len(test_loader): .6f}, Accuracy: {sum_acc/len(test_loader): .6f}')
    print(f'Time: {time.time()-since: .1f} s')

# save the model when training's done
torch.save(model.state_dict(), './logistic.pth')
