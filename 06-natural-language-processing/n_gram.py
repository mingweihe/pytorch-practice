import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigram = [((test_sentence[i], test_sentence[i+1]), test_sentence[i+2])
    for i in range(len(test_sentence)-2)]

vocab = set(test_sentence)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
class NgramModel(nn.Module):
    def __init__(self, vocab_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.n_word = vocab_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(1, -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x

model = NgramModel(len(word_to_idx), CONTEXT_SIZE, 100)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for epoch in range(1, 101):
    print(f'epoch: {epoch}')
    print('*'*20)
    sum_loss = 0
    for word, label in trigram:
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # forward
        out = model(word)
        loss = criterion(out, label)
        sum_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Loss: {sum_loss/len(word_to_idx):.6f}')
word, label = trigram[3]
word = Variable(torch.LongTensor([word_to_idx[w] for w in word]))
out = model(word)
_, pred = torch.max(out, 1)
pred = idx_to_word[pred.item()]
print(f'Real word is {label}, predict word is {pred}')
