import torch
from torch.autograd import Variable
from torch import nn, optim
from data_utils import Corpus

seq_length = 30
train_file = 'train.txt'

train_corpus = Corpus()
valid_corpus = Corpus()
test_corpus = Corpus()

train_id = train_corpus.get_data(train_file)

print(train_id[0])

vocab_size = len(train_corpus.dic)
num_batches = train_id.size(1) // seq_length
num_epochs = 5
learning_rate = 1e-3

class languagemodel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(languagemodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        x, hi = self.lstm(x, h)
        b, s, h = x.size()
        x = x.contiguous().view(b*s, h)
        x = self.linear(x)
        return x, hi

model = languagemodel(vocab_size, 128, 1024, 1)
use_gpu = torch.cuda.is_available()
if use_gpu: model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def detach(states):
    if use_gpu: return [Variable(state.data).cuda() for state in states]
    return [Variable(state.data) for state in states]

for epoch in range(1, num_epochs+1):
    print('*' * 20)
    print(f'epoch {epoch}')
    sum_loss = 0
    if use_gpu: states = (Variable(torch.zeros(1,20,1024)).cuda(), Variable(torch.zeros(1, 20, 1024)).cuda())
    else: states = (Variable(torch.zeros(1,20,1024)), Variable(torch.zeros(1, 20, 1024)))
    for i in range(0, train_id.size(1)-2*seq_length, seq_length):
        input_x = train_id[:, i:(i+seq_length)]
        label = train_id[:, (i+seq_length):(i+2*seq_length)]
        if use_gpu:
            input_x = Variable(input_x).cuda()
            label = Variable(label).cuda()
        else:
            input_x = Variable(input_x)
            label = Variable(label)
        label = label.reshape(label.size(0)*label.size(1), 1)
        # forward 
        states = detach(states)
        out, states = model(input_x, states)
        loss = criterion(out, label.view(-1))
        sum_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimizer.step()

        step = (i+1) // seq_length
        print(f'Epoch [{epoch}/{num_epochs}], Step[{step}/{num_batches}], Loss: {loss.item()}')
