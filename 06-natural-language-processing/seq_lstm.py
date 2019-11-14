import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
import string

training_data = [("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])]

word_to_idx = {}
tag_to_idx = {}
idx_to_tag = {}
for context, tag in training_data:
    for word in context:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    for label in tag:
        if label not in tag_to_idx:
            tag_to_idx[label] = len(tag_to_idx)
            idx_to_tag[tag_to_idx[label]] = label

alphabet = string.ascii_lowercase
character_to_idx = {}
for i in range(len(alphabet)):
    character_to_idx[alphabet[i]] = i
use_gpu = torch.cuda.is_available()
class CharLSTM(nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()
        self.char_embedding = nn.Embedding(n_char, char_dim)
        self.char_lstm = nn.LSTM(char_dim, char_hidden, batch_first=True)
    
    def forward(self, x):
        x = self.char_embedding(x)
        _, h = self.char_lstm(x)
        return h[0]

class LSTMTagger(nn.Module):
    def __init__(self, n_word, n_char, char_dim, n_dim, char_hidden, n_hidden, n_tag):
        super(LSTMTagger, self).__init__()
        self.word_embedding = nn.Embedding(n_word, n_dim)
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)
        self.lstm = nn.LSTM(n_dim + char_hidden, n_hidden, batch_first=True)
        self.linear1 = nn.Linear(n_hidden, n_tag)
    
    def forward(self, x, word):
        char = torch.FloatTensor()
        for w in word:
            char_list = []
            for letter in w:
                char_list.append(character_to_idx[letter.lower()])
            char_list = torch.LongTensor(char_list)
            char_list = char_list.unsqueeze(0)
            if use_gpu: tempchar = self.char_lstm(Variable(char_list).cuda())
            else: tempchar = self.char_lstm(Variable(char_list))
            tempchar = tempchar.squeeze(0)
            char = torch.cat((char, tempchar.cpu().data), 0)
        if use_gpu: char = char.cuda()
        char = Variable(char)
        x = self.word_embedding(x)
        x = torch.cat((x, char), 1)
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        x = self.linear1(x)
        x = F.log_softmax(x, dim=1)
        return x

model = LSTMTagger(len(word_to_idx), len(character_to_idx), 10, 100, 50, 128, len(tag_to_idx))
if use_gpu: model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

def make_sequence(x, dic):
    idx = [dic[i] for i in x]
    idx = Variable(torch.LongTensor(idx))
    return idx

for epoch in range(1, 301):
    print('*'*20)
    print(f'epoch {epoch}')
    sum_loss = 0
    for word, tag in training_data:
        word_list = make_sequence(word, word_to_idx)
        tag = make_sequence(tag, tag_to_idx)
        if use_gpu: word_list, tag = word_list.cuda(), tag.cuda()
        # forward
        out = model(word_list, word)
        loss = criterion(out, tag)
        sum_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Loss: {sum_loss/len(training_data)}\n')
test_setence = 'Everybody ate the apple'
input = make_sequence(test_setence.split(), word_to_idx)
if use_gpu: input = input.cuda()
out = model(input, test_setence.split())
_, pred = torch.max(out, 1)
print(' '.join(map(idx_to_tag.get, pred.numpy())))
torch.save(model.state_dict(), 'model.pth')