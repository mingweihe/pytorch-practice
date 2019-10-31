import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# from sklearn.utils import check_random_state
# n = 100
# x = np.arange(n)
# rs = check_random_state(0)
# y = rs.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))

# x_train = np.array(x.reshape(-1, 1), dtype=np.float32)
# y_train = np.array(y.reshape(-1, 1), dtype=np.float32)

# numpy array to tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# linear regression model
class linear_regression(nn.Module):
    def __init__(self):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(1, 1) # input and output is 1 dimension
        
    def forward(self, x):
        out = self.linear(x)
        return out
    
model = linear_regression()
# define loss and optimization function
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# start training
num_epochs = 1000
epoch = 0
loss = float('inf')
for epoch in range(1, 21):
    inputs = x_train
    target = y_train
    
    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch[{epoch}/{num_epochs}], loss: {loss.item(): .6f}')
# eval mode - prevent batchnorm and dropout operations   
model.eval()
with torch.no_grad():
    predictions = model(x_train)
predictions = predictions.data.numpy()

fig = plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data') # ro for red circles
plt.plot(x_train.numpy(), predictions, 'o-', color='#1f77b4', label='Fitting Line')
# show diagram
plt.legend()
plt.show()

# # save the model to file
PATH = './linear.pth'

# model parameters only
# torch.save(model.state_dict(), PATH)

# parameters needed for resuming training
torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, PATH)

# load the model and continuously train for another 999 epochs
model = linear_regression()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f'last epoch is {epoch}, loss is {loss}')
criterion = nn.MSELoss()

# training mode
model.train()
# eval mode - prevent batchnorm and dropout operations
# model.eval()

for epoch in range(epoch+1, num_epochs+1):
    inputs = x_train
    target = y_train
    
    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch[{epoch}/{num_epochs}], loss: {loss.item(): .6f}')
# eval mode - prevent batchnorm and dropout operations
model.eval()
with torch.no_grad():
    predictions = model(x_train)
predictions = predictions.data.numpy()

fig = plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data') # ro for red circles
plt.plot(x_train.numpy(), predictions, 'o-', color='#1f77b4', label='Fitting Line')

plt.legend()
plt.show()

# -------------------------------------------------
# model save load another way (entire model) -- not recommand

# torch.save(model, './entire_model.pth')
# model = torch.load('./entire_model.pth')
