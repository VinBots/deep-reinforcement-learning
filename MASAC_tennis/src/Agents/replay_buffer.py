# Import libraries
import random
from collections import namedtuple, deque
import torch
import numpy as np

class Centralized_ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed,num_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.num_agents = num_agents
        self.memory = deque(maxlen=buffer_size)
        self.all_fields_names = ["states", "actions", "rewards", "next_states", "dones"]
        self.record_length = len(self.all_fields_names)
        self.experience = namedtuple("Experience", field_names=self.all_fields_names)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        """
        ====
        Data Structure.

            Each experience is made of a list of states (s), a list of actions (a),
            a list of rewards (r), a list of next_states(s') and a list of dones (d)

            The i-th element of each list of s,a,r,s' records the s,a,r,s for agent of index i
            For example, experience of agent2 =
            (experience.states[2],experience.actions[2],experience.rewards[2],experience.next_states[2]
        """

    def add(self, states, actions, rewards, next_states, dones):

        """Add a new experience to memory."""
        e=self.experience._make((states, actions, rewards, next_states, dones))
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        """Returns torch tensors"""

        experiences = random.sample(self.memory, k=self.batch_size)

        all_states = tuple(torch.from_numpy(np.vstack([e.states[i] for e in experiences if e is not None]))\
                                   .float().to(self.device) for i in range (self.num_agents))

        all_actions = tuple(torch.from_numpy(np.vstack([e.actions[i] for e in experiences if e is not None]))\
                                   .float().to(self.device) for i in range (self.num_agents))

        all_rewards = tuple(torch.from_numpy(np.vstack([e.rewards[i] for e in experiences if e is not None]))\
                                   .float().to(self.device) for i in range (self.num_agents))

        all_next_states = tuple(torch.from_numpy(np.vstack([e.next_states[i] for e in experiences if e is not None]))\
                                   .float().to(self.device) for i in range (self.num_agents))

        all_dones = tuple(torch.from_numpy(np.vstack([e.dones[i] for e in experiences if e is not None]).astype(np.uint8))\
                          .float().to(self.device) for i in range(self.num_agents))

        return (all_states, all_actions, all_rewards, all_next_states, all_dones)

    def buffer_len(self):
        """Return the current size of internal memory."""
        return len(self.memory)
