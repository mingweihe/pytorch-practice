import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from logger2 import Logger
from torchsummary import summary
import datetime

# hyper parameters
batch_size = 128
learning_rate = 1e-2
num_epochs = 20

def to_np(x): return x.cpu().data.numpy()

# download MNIST digit data set
train_set = datasets.MNIST(
    root='../data', train=True, transform=transforms.ToTensor(), download=True
)
test_set = datasets.MNIST(
    root='../data', train=False, transform=transforms.ToTensor()
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# define a convolutional neural network model
class cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_class)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = cnn(1, 10)
use_gpu = torch.cuda.is_available()
summary(model, (1, 28, 28))

if use_gpu: model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
logger = Logger('./logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# start training
for epoch in range(1, num_epochs+1):
    print('*' * 20)
    print(f'epoch {epoch}')
    sum_loss = .0
    sum_acc = .0
    model.train()
    for i, data in enumerate(train_loader, 1):
        img, label = data

        if use_gpu: img, label = img.cuda(), label.cuda()
        # forward propagation
        out = model(img)
        loss = criterion(out, label)
        sum_loss += loss.item()
        _, pred = torch.max(out, 1)
        cur_acc = (pred==label).float().mean()
        sum_acc += cur_acc
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # logging
        # step here means the n-th updating of all epochs
        step = (epoch-1) * len(train_loader) + i
        # 1. log the scalar values
        info = {'loss': loss.item(), 'accuracy': cur_acc.numpy()}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)
        # 2. log values and gradients of parameters (histogram)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histogram_summary(tag, to_np(value), step)
            logger.histogram_summary(tag + '/grad', to_np(value.grad), step)

        # 3. log the images
        info = {'images': [to_np(img.view(-1, 28, 28)[:10]), to_np(pred[:10])]}
        for tag, vals in info.items():
            logger.image_summary(tag, vals, step)
        if i % 300 == 0:
            print(f'[{epoch}/{num_epochs}] Loss: {sum_loss/i: .6f} \
                Acc: {sum_acc/i: .6f}')
    print(f'Finish {epoch} epoch, Loss: {sum_loss/i: .6f}, \
        Acc: {sum_acc/i: .6f}')

    model.eval()
    sum_loss = .0
    sum_acc = .0
    for data in test_loader:
        img, label = data
        if use_gpu: img, label = img.cuda(), label.cuda()
        with torch.no_grad():
            out = model(img)
        loss = criterion(out, label)
        sum_loss += loss.item()
        _, pred = torch.max(out, 1)
        sum_acc += (pred==label).float().mean()
    print(f'Test loss: {sum_loss/len(test_loader): .6f} \
        Accuray: {sum_acc/len(test_loader): .6f} \n')
# save the model
torch.save(model.state_dict(), './cnn.pth')
