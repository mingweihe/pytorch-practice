import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchsummary import summary

# hyper parameters
batch_size = 100
learning_rate = 1e-3
num_epochs = 1

# download MNIST digit data set
train_set = datasets.MNIST(
    root='../data', train=True, transform=transforms.ToTensor(), download=True
)
test_set = datasets.MNIST(
    root='../data', train=False, transform=transforms.ToTensor()
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# define a rnn model
class rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(rnn, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)
        # self.n_layer = n_layer
        # self.hidden_dim = hidden_dim
        

    def forward(self, x):
        # h0 = torch.randn(self.n_layer, x.size(0), self.hidden_dim)
        # c0 = torch.randn(self.n_layer, x.size(0), self.hidden_dim)
        # if use_gpu: h0, c0 = h0.cuda(), c0.cuda()
        # x, _ = self.lstm(x, (h0, c0))
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x

model = rnn(28, 128, 2, 10)
use_gpu = torch.cuda.is_available()
if use_gpu: model = model.cuda()
# replace official torchsummary with https://github.com/Bond-SYSU/pytorch-summary
# 1. pip uninstall torchsummary
# 2. cd to project folder, then python setup.py install
# otherwise, remove this line
summary(model, (1, 28))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# start training
for epoch in range(1, num_epochs+1):
    print('*' * 20)
    print(f'epoch {epoch}')
    sum_loss = .0
    sum_acc = .0
    model.train()
    for i, data in enumerate(train_loader, 1):
        img, label = data
        b, c, h, w = img.size()
        assert c == 1, 'channel must be 1'
        img = img.squeeze(1)
        if use_gpu: img, label = img.cuda(), label.cuda()
        # forward propagation
        out = model(img)
        loss = criterion(out, label)
        sum_loss += loss.item()
        _, pred = torch.max(out, 1)
        sum_acc += (pred==label).float().mean()
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 300 == 0:
            print(f'[{epoch}/{num_epochs} - step: {i}] Loss: {sum_loss/i: .6f} \
                Acc: {sum_acc/i: .6f}')
    print(f'Finish {epoch} epoch, Loss: {sum_loss/i: .6f} \
        Acc: {sum_acc/i: .6f}')
    model.eval()
    sum_loss = .0
    sum_acc = .0
    for data in test_loader:
        img, label = data
        b, c, h, w = img.size()
        assert c == 1, 'channel must be 1'
        img = img.squeeze(1)
        if use_gpu: img, label = img.cuda(), label.cuda()
        with torch.no_grad(): out = model(img)
        loss = criterion(out, label)
        sum_loss += loss.item()
        _, pred = torch.max(out, 1)
        sum_acc += (pred==label).float().mean()
    print(f'Test loss: {sum_loss/len(test_loader): .6f}, \
        Acc: {sum_acc/len(test_loader): .6f}\n')

# save the model
torch.save(model.state_dict(), './rnn.pth')