import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary

# hyper parameters
batch_size = 64
learning_rate = 1e-2
num_epochs = 50
use_gpu = torch.cuda.is_available()

# download Fashion MNIST data set
train_set = datasets.FashionMNIST(
    root='../data', train=True, transform=transforms.ToTensor(), download=True
)

test_set = datasets.FashionMNIST(
    root='../data', train=False, transform=transforms.ToTensor()
)

# wrap into batches, and shuffle
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# define a neural network model
class neural_network(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(neural_network, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = neural_network(28*28, 300, 100, 10)
if use_gpu: model = model.cuda()
summary(model, (1, 28*28))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# start training
for epoch in range(1, num_epochs+1):
    print('*' * 20)
    print(f'epoch {epoch}')
    sum_loss = .0
    sum_acc = .0
    model.train()
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
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
            print(f'[{epoch}/{num_epochs}, step: {i}] Loss: {sum_loss/i: .6f}, \
                Acc: {sum_acc/i: .6f}')
    print(f'Finish {epoch} epoch, Loss: {sum_loss/i: .6f}, Acc: {sum_acc/i: .6f}')

    # evaluation after each epoch
    model.eval()
    sum_loss = .0
    sum_acc = .0
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()

        with torch.no_grad():
            out = model(img)
            loss = criterion(out, label)

        sum_loss += loss.item()
        _, pred = torch.max(out, 1)
        sum_acc += (label==pred).float().mean()
    print(f'Test loss: {sum_loss/len(test_loader): .6f}, \
        Acc: {sum_acc/len(test_loader): .6f}')

# save the model
torch.save(model.state_dict(), './neural_network.pth')
