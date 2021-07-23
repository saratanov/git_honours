"""
Minimal character-level Pytorch LSTM model. Written by Jo Plested based on vanilla RNN by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# data load dataset
data = open('textData.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }


# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 50 # number of steps to unroll the RNN for
learning_rate = 1e-1

class _charLSTM(nn.Module):
	def __init__(self):
		super(_charLSTM, self).__init__()
		self.LSTM = nn.LSTM(len(chars), hidden_size, 1)
		self.FC = nn.Linear(hidden_size, len(chars))

	def forward(self, input, hprev, cprev, seq_len):
		input = input.view(input.data.shape[0], 1, input.data.shape[1])
		output, hc = self.LSTM(input, (hprev, cprev))
		output = self.FC(output)
		return output.view(seq_len, len(chars)), hc


charLSTM = _charLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(charLSTM.parameters(), lr=learning_rate)


def sample(h, c, seed_ix, n):
  """ 
  sample a sequence of outputs from the model 
  h and c are memory states, 
  seed_ix is seed letter for first time step,
  n is number of outputs to sample
  """
  letter = torch.zeros(1, len(chars))
  letter.data[0][seed_ix] = 1
  letters = []
  """
  sample letters one at a time so output at previous step 
  becomes input for next step
  """
  for t in range(n):
    y, hc = charLSTM(letter, h, c, 1)
    p = F.softmax(y,1)
    l = np.random.choice(range(vocab_size), p=p.data.numpy().ravel())
    letters.append(l)
    letter = torch.zeros(1, len(chars))
    letter.data[0][l] = 1
    h = hc[0]
    c = hc[1]
  return letters

p,n = 0,0
smooth_loss = -np.log(1.0/vocab_size)# loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = torch.zeros(1, 1, hidden_size) # reset RNN memory
    cprev = torch.zeros(1, 1, hidden_size)
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
  targets = torch.from_numpy(np.array(targets))

  # sample from the model now and then
  if n % 1000 == 0:
    sample_ix = sample(hprev, cprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print('----\n %s \n----' % (txt, ))
 
  # forward seq_length characters through the net
  input = torch.zeros(seq_length, len(chars))
  for i,inp in enumerate(input):
    inp[inputs[i]]=1
  charLSTM.zero_grad()
  output, hcprev = charLSTM(input, hprev, cprev, seq_length)
  hprev = hcprev[0].detach()
  cprev = hcprev[1].detach()
  
  # get loss and gradients and adjust the weights
  err = criterion(output, targets)
  err.backward()
  optimizer.step()
  smooth_loss = smooth_loss * 0.999 + err.detach().numpy() * 0.001
  if n % 1000 == 0: 
    print('iter %d, loss: %f' % (n, smooth_loss)) # print progress

  p += seq_length # move data pointer
  n += 1 # iteration counter 
