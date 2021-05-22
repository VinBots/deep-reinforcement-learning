# Import libraries
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class Soft_Policy_Network(nn.Module):

    def __init__(self, input_size, h1_size, h2_size, output_mean_size, output_std_size):
        super(Soft_Policy_Network, self).__init__()

        # state, hidden layer, action sizes
        self.input_size = input_size
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.output_mean_size = output_mean_size
        self.output_std_size = output_std_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # define layers
        self.fc1 = nn.Linear(self.input_size, self.h1_size)
        self.fc2 = nn.Linear(self.h1_size, self.h2_size)
        self.fc3_mean = nn.Linear(self.h2_size, self.output_mean_size)
        self.fc3_log_std = nn.Linear(self.h2_size, self.output_std_size)
        #initialize weights
        init_w = 3e-3
        self.fc3_mean.weight.data.uniform_(-init_w,init_w)
        self.fc3_mean.bias.data.uniform_(-init_w,init_w)
        self.fc3_log_std.weight.data.uniform_(-init_w,init_w)
        self.fc3_log_std.bias.data.uniform_(-init_w,init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc3_mean(x) #values of the action should be between -1 and 1 so this is not the mean of the action value
        log_std = self.fc3_log_std(x)
        log_std_min = -20
        log_std_max = 0
        log_std = torch.clamp(log_std,log_std_min, log_std_max)
        return mean,log_std

    def sample (self,state,epsilon = 1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal (mean,std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1,keepdim=True)
        return action, log_pi

    def get_action(self, state,deterministic):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)

        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            z = normal.sample() #sample an action from a normal distribution with (mean,std)
            action = torch.tanh(z) #squeeze the value between -1 and 1

        action = action.cpu().detach().squeeze(0).numpy()
        return self.rescale_action(action)

    def rescale_action(self, action):
        action_range=[-1,1]
        return action * (action_range[1] - action_range[0]) / 2.0 +\
            (action_range[1] + action_range[0]) / 2.0
