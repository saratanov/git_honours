"""
This script shows an example of building a linear regression
model for y = 3x + 3
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Hyper Parameters
input_size = 1
num_classes = 1
num_epochs = 500
learning_rate = 0.01


# create some sample data for y = 3x + 3
# create 100 sample data in [-1, 1]
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# create 100 sample noisy sample output
y = x * 3 + 3 + 0.5*torch.rand(x.size())

# plot sample x and y
# plt.scatter(x.data.numpy(), y.data.numpy())


# define our regression model
class Regression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Regression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


reg_model = Regression(1, 1)

# define loss function
# Softmax is internally computed in nn.CrossEntropyLoss.
loss_func = nn.MSELoss()

# define optimiser
optimiser = torch.optim.SGD(reg_model.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []

# turn plot on
plt.ion()

for t in range(num_epochs):
    # pass input x and get prediction
    prediction = reg_model(x)

    # calculate loss
    loss = loss_func(prediction, y)

    # clear gradients for next train
    optimiser.zero_grad()

    # perform backward pass
    loss.backward()

    # call the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

