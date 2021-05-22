# Import libraries
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from unityagents import UnityEnvironment
import Agents.SAC
import Agents.replay_buffer

class Multiple_Agents:

    def __init__(self, game,replay_buffer_size=50000,batch_size=256,load_mode=False,\
                 save_mode=True, episods_before_update=10):

        self.game = game
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.load_mode = load_mode
        self.save_mode=save_mode
        self.episods_before_update = episods_before_update

        self.episod_index = 0
        self.num_agents = self.game.num_agents

        # Creation of the agents
        self.agents = [Agents.SAC.SAC_Agent(index=i,state_dim = self.game.state_dim,action_dim = self.game.action_dim) for i in range(self.num_agents)]
        # Creation of the replay buffer and game scores
        self.replay_buffer = Agents.replay_buffer.Centralized_ReplayBuffer(self.replay_buffer_size, self.batch_size, 1,self.num_agents)

        # Loading of existing agents
        if load_mode:
            file_name = "checkpoint_masac_"
            for agent in self.agents:
                critic1_path = file_name + "critic1_Agent" + str(agent.index)+".pth"
                critic2_path = file_name + "critic2_Agent" + str(agent.index)+".pth"
                actor_path = file_name + "actor_Agent" + str(agent.index)+".pth"
                agent.load_weights(critic1_path, critic2_path,actor_path)

    def get_action(self,states,deterministic):
        actions = []
        for state,agent in zip(states,self.agents):
            action = agent.get_action(state,deterministic)
            actions.append(action)
        return actions

    def update_per_step(self):
        self.episod_index+=1

        if self.update_gateway():
            # Sample a batch of experiences from the centralized buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample()

            # Update critics and actors
            for agent in self.agents:
                agent.update(states, actions, rewards, next_states, dones,self.agents)

    def update_gateway(self):
        test1 = self.replay_buffer.buffer_len() >=self.batch_size
        test2 = self.episod_index > self.episods_before_update
        gateway = test1 and test2
        return gateway

    def save_weights(self):
        if self.save_mode:
            for agent in self.agents:
                agent.save_weights()

    def training(self,num_epochs,training_mode=True):
        # Launch the environment

        env = UnityEnvironment(file_name="Tennis.app")
        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        actions=np.zeros(self.game.num_agents,)

        for each_iteration in range(num_epochs):

            # Initialization
            actions=np.zeros(self.game.num_agents,)

            env_info=env.reset(train_mode=training_mode)[brain_name]

            states = env_info.vector_observations

            scores = np.zeros(self.game.num_agents)

            for each_environment_step in range(self.game.num_steps_per_epoch):
                #interacts with the environment by sampling actions and collect next_states, rewards and status
                actions = self.get_action(states,deterministic = not(training_mode))
                env_info = env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                #computes scores of all the agents
                scores += rewards

                #Store the transition in the replay buffer
                self.replay_buffer.add(states,actions,rewards,next_states,dones)

                #updates the critic and actor
                if training_mode:
                    self.update_per_step()

                states = next_states

                if np.any(dones) or each_environment_step == self.game.num_steps_per_epoch - 1:
                    break

            episod_score = self.game.compute_episod_score(scores)
            self.game.game_score.update_scores(episod_score)
            self.game.game_score.display_score()

            if self.game.test_for_ending():
                print("\nGame {} solved in {:d} episodes!".format(self.game.name, each_iteration))
                self.save_weights()
                env.close()
                break
