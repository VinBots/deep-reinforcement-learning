# Import libraries
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

class Soft_Q_Network(nn.Module):

    def __init__(self, input_size, h1_size, h2_size, output_size):
        super(Soft_Q_Network, self).__init__()

        # state, hidden layer, action sizes
        self.input_size = input_size
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.output_size = output_size

        # define layers
        self.fc1 = nn.Linear(self.input_size, self.h1_size)
        self.fc2 = nn.Linear(self.h1_size, self.h2_size)
        self.fc3 = nn.Linear(self.h2_size, self.output_size)

        #initialize weights
        init_w = 3e-3
        self.fc3.weight.data.uniform_(-init_w,init_w)
        self.fc3.bias.data.uniform_(-init_w,init_w)

    def forward(self, state,action):
        x = torch.cat([state,action],1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
