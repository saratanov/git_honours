# -*- coding: utf-8 -*-
"""
Adapted from Reinforcement Learning (DQN) tutorial
=====================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_
by Jo Plested
"""

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical

outf = './Output'
previous_state_dict = ''


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNet(nn.Module):

    def __init__(self, h, w, outputs):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w)), kernel_size=3), kernel_size=3, stride=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h)), kernel_size=3), kernel_size=3, stride=1)
        linear_input_size = convw * convh
        self.head = nn.Linear(linear_input_size, outputs)


        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.conv4(x))
        x = self.head(x.view(x.size(0), -1))
        return F.softmax(x, dim=1)


######################################################################
# Input extraction
# ^^^^^^^^^^^^^^^^
#
# The code below are utilities for extracting and processing rendered
# images from the environment. It uses the ``torchvision`` package, which
# makes it easy to compose image transforms. Once you run the cell it will
# display an example patch that it extracted.
#

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
          interpolation='none')
plt.title('Example extracted screen')
plt.show()


BATCH_SIZE = 128
GAMMA = 0.9999


# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = PolicyNet(screen_height, screen_width, n_actions).to(device)
if previous_state_dict != '':
    policy_net.load_state_dict(torch.load(previous_state_dict, map_location='cpu'))

optimizer = optim.RMSprop(policy_net.parameters(), lr = 0.001)


def select_action(state):
    state = state.float()
    probs = policy_net(state)
    m = Categorical(probs)
    action = m.sample()
    policy_net.saved_log_probs.append(m.log_prob(action))
    return action.data[0]


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy_net.rewards[::-1]:
        R = r + GAMMA * R        
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for log_prob, reward in zip(policy_net.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy_net.rewards[:]
    del policy_net.saved_log_probs[:]

episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



running_reward = 10
num_episodes = 50000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    #print(state)
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        policy_net.rewards.append(reward)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Move to the next state
        state = next_state

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    
    running_reward = running_reward * 0.99 + t * 0.01
    finish_episode()
   
    if i_episode % 50 == 0:
        for param_group in optimizer.param_groups:
         print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
         #print(i_episode)
         print(param_group['lr'])

        # do checkpointing
        torch.save(policy_net.state_dict(), 'policy_net_epoch_%d.pth' % (i_episode))


print('Complete')
#env.render(close=True)
env.close()
plt.ioff()
plt.show()
