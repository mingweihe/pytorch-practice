import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

"""
The continuous bag-of-words model (CBOW) is frequently used in NLP deep learning.
It is a model that tries to predict words given the context of a few words after
the target word. This is distinct from language modeling, since SBOW is not 
sequential and does not have to be probabilistic. Typically, CBOW is used to 
queickly train word embeddings, and thes embeddings are used to initialize the 
embeddings of some more complicated model. Usually, this is referred to as
pretraining embeddings. It almost always helps performance a couple of percent.
"""

CONTEXT_SIZE = 2 # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
word_to_idx = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(CONTEXT_SIZE, len(raw_text)-CONTEXT_SIZE):
    context = [raw_text[i-2], raw_text[i-1], raw_text[i+1], raw_text[i+2]]
    target = raw_text[i]
    data.append((context, target))

class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)
        self.project = nn.Linear(n_dim, n_dim, bias=False)
        self.linear1 = nn.Linear(n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.project(x)
        x = torch.sum(x, 0, keepdim=True)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x)
        return x

model = CBOW(len(word_to_idx), 100, CONTEXT_SIZE)
use_gpu = torch.cuda.is_available()
if use_gpu: mode = model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(1, 101):
    print(f'epoch {epoch}')
    print('*'*20)
    sum_loss = 0
    for word in data:
        context, target = word
        context = Variable(torch.LongTensor([word_to_idx[i] for i in context]))
        target = Variable(torch.LongTensor([word_to_idx[target]]))
        if use_gpu: context, target = context.cuda(), target.cuda()
        # forward
        out = model(context)
        loss = criterion(out, target)
        sum_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'loss: {sum_loss/len(data)}')
