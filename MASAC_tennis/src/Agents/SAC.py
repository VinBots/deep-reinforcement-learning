# Import libraries
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import Networks.soft_policy_network
import Networks.soft_Q_network

class SAC_Agent:

    def __init__(self, index,state_dim = 24 ,action_dim = 2,layer_size = 128, qf_lr = 0.0006, \
                 policy_lr=0.0003,a_lr = 0.0006,auto_entropy_tuning=True,soft_target_tau =0.02, discount = 0.99):

        self.index= index #starts at 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.layer_size = layer_size
        self.qf_lr = qf_lr
        self.policy_lr = policy_lr
        self.a_lr = a_lr
        self.auto_entropy_tuning = auto_entropy_tuning
        self.soft_target_tau = soft_target_tau
        self.discount = discount

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.qf1 = Networks.soft_Q_network.Soft_Q_Network(
            input_size = 2*(state_dim + action_dim),
            h1_size = layer_size,
            h2_size = layer_size,
            output_size=1

        ).to(self.device)

        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=qf_lr)

        self.qf2 = Networks.soft_Q_network.Soft_Q_Network(
            input_size = 2 * (state_dim + action_dim),
            h1_size = layer_size,
            h2_size = layer_size,
            output_size=1
        ).to(self.device)

        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=qf_lr)

        self.target_qf1 = Networks.soft_Q_network.Soft_Q_Network(
            input_size = 2*(state_dim + action_dim),
            h1_size = layer_size,
            h2_size = layer_size,
            output_size=1
        ).to(self.device)

        self.target_qf2 = Networks.soft_Q_network.Soft_Q_Network(
            input_size = 2*(state_dim + action_dim),
            h1_size = layer_size,
            h2_size = layer_size,
            output_size=1
        ).to(self.device)

        self.policy = Networks.soft_policy_network.Soft_Policy_Network(
            input_size = state_dim,
            h1_size = layer_size,
            h2_size = layer_size,
            output_mean_size = action_dim,
            output_std_size = action_dim
        ).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)

        if self.auto_entropy_tuning:
            self.target_entropy = -0.000
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.log_alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

        # copy parameters of qf1/qf2 to target_qf1/target_qf2
        for target_params, params in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            target_params.data.copy_(params)

        for target_params, params in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            target_params.data.copy_(params)

    def get_action(self, state,deterministic):
        return self.policy.get_action(state,deterministic)

    def save_weights(self):
        torch.save(self.qf1.state_dict(), "checkpoint_masac_critic1_Agent"+str(self.index)+".pth")
        torch.save(self.qf2.state_dict(), "checkpoint_masac_critic2_Agent"+str(self.index)+".pth")
        torch.save(self.policy.state_dict(), "checkpoint_masac_actor_Agent"+str(self.index)+".pth")

    def load_weights(self,critic1_path,critic2_path,actor_path):
        self.qf1.load_state_dict(torch.load(critic1_path))
        self.qf1.eval()
        print ("Models: {} loaded...".format(critic1_path))
        self.qf2.load_state_dict(torch.load(critic2_path))
        self.qf2.eval()
        print ("Models: {} loaded...".format(critic2_path))
        self.policy.load_state_dict(torch.load(actor_path))
        self.policy.eval()
        print ("Models: {} loaded...".format(actor_path))

    def update(self,states, actions, rewards, next_states, dones,ma_agents):
        num_agents = len(ma_agents)
        #concatenate states, actions, next_states and next_actions
        all_states = torch.cat(tuple(states[i] for i in range(num_agents)),dim=1)
        all_actions = torch.cat(tuple(actions[i] for i in range(num_agents)),dim=1)
        all_next_states = torch.cat(tuple(next_states[i] for i in range(num_agents)),dim=1)
        local_rewards = rewards[self.index]
        local_dones = dones[self.index]

        ###### POLICY EVALUATION STEP ######

        #Update the collective Q-function parameters for the agent
        #Predict next actions after next_states for all the agents
        next_actions=[]
        next_log_pis = []

        for agent in ma_agents:
            local_next_actions, local_next_log_pis = agent.policy.sample(next_states[agent.index])
            next_actions.append(local_next_actions)
            next_log_pis.append (local_next_log_pis)

        all_next_actions = torch.cat(tuple(next_actions[i] for i in range(num_agents)),dim=1)

        next_qf1 = self.target_qf1.forward(all_next_states,all_next_actions)
        next_qf2 = self.target_qf2.forward(all_next_states,all_next_actions)
        next_q_target = torch.min(next_qf1,next_qf2) - self.alpha * next_log_pis[self.index]
        expected_q = local_rewards + (1 - local_dones) * self.discount * next_q_target
        curr_qf1 = self.qf1.forward(all_states,all_actions)
        curr_qf2 = self.qf2.forward(all_states,all_actions)
        qf1_loss = F.mse_loss(curr_qf1, expected_q.detach())
        qf2_loss = F.mse_loss(curr_qf2, expected_q.detach())

        #Update critic1 weights
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        #Update critic2 weights
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        ###### POLICY IMPROVEMENT STEP ######
        #Predict new actions after the current state
        new_actions = []
        new_log_pis = []

        for agent in ma_agents:
            local_new_actions, local_new_log_pis = agent.policy.sample(states[agent.index])
            new_actions.append(local_new_actions)
            new_log_pis.append (local_new_log_pis)

        all_new_actions = torch.cat(tuple(new_actions[i] for i in range(num_agents)),dim=1)
        local_log_pis = new_log_pis[self.index]

        min_q = torch.min(self.qf1.forward(all_states, all_new_actions),
                          self.qf2.forward(all_states, all_new_actions))

        policy_loss = (self.alpha * local_log_pis - min_q).mean()

        #Update actor weights
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        #print ("Policy of agents {} updated".format(self.index))

        #Update target network weights at every iteration
        for target_params, params in zip(self.target_qf1.parameters(), self.qf1.parameters()):
            target_params.data.copy_(self.soft_target_tau * params + (1 - self.soft_target_tau) * target_params)

        for target_params, params in zip(self.target_qf2.parameters(), self.qf2.parameters()):
            target_params.data.copy_(self.soft_target_tau * params + (1 - self.soft_target_tau) * target_params)

        #Adjust entropy temperature
        if self.auto_entropy_tuning:
            self.log_alpha_optim.zero_grad()
            alpha_loss = (self.log_alpha * (-local_log_pis - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optim.step()
            self.alpha = self.log_alpha.exp()
