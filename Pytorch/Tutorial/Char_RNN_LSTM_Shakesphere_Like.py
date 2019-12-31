import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import random
import unidecode
import string
import re
import time, math

'''Hyperparameter'''

num_epochs = 10000
print_every = 100
plot_every = 10
chunk_len = 200
embedding_size = 150
hidden_size = 100
batch_size = 1
num_layers = 1
lr = 0.002


'''Data Processing'''

all_characters = string.printable
n_characters = len(all_characters)

file = unidecode.unidecode(open('./data/shakespeare.txt').read())
file_len = len(file)

def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor).cuda()

def random_training_set():
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:])
    return inp, target

class LSTM(nn.Module):

    def __init__(self, input_size, embedding_size,hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers

        self.encoder = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first = True)
        self.decoder = nn.Linear(hidden_size,output_size)

    def forward(self, input, hidden, cell):
        out = self.encoder(input.view(batch_size, -1))
        out, (hidden, cell) = self.lstm(out, (hidden, cell))
        out = self.decoder(out.view(batch_size, -1))

        return out,hidden,cell

    def init_hidden(self):

        hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()
        cell = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()

        return hidden, cell


model = LSTM(n_characters, embedding_size, hidden_size, n_characters, num_layers).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

def test():
    start_str = "b"
    inp = char_tensor(start_str)
    hidden,cell = model.init_hidden()
    x = inp

    print(start_str,end="")
    for i in range(200):
        output,hidden,cell = model(x,hidden,cell)

        output_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_characters[top_i]

        print(predicted_char,end="")

        x = char_tensor(predicted_char)


'''Training'''

for i in range(num_epochs):

    input, target = random_training_set()
    hidden, cell = model.init_hidden()
    loss = 0
    optimizer.zero_grad()

    for j in range(chunk_len -1):
        x = input[j]
        y_ = target[j]
        y, hidden, cell = model(x, hidden, cell)
        loss += criterion(y, y_.unsqueeze(0))

    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print("\n",loss/chunk_len,"n")
        test()
        print("\n\n")
