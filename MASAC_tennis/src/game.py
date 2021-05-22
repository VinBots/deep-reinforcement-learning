# Import libraries
import numpy as np
import Game_Utils.game_score

class Game:

    def __init__(self, name, solve_score,state_dim,action_dim,num_agents,num_steps_per_epoch=1000):
        self.name = name
        self.solve_score = solve_score
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.num_steps_per_epoch = num_steps_per_epoch
        self.game_score = Game_Utils.game_score.GameScore()

    def compute_episod_score(self, scores):
        return np.max(scores)

    def test_for_ending(self):
        test1 = (self.game_score.last_moving_average>=self.solve_score)
        test2 = (len(self.game_score.all_scores) >=100)
        end_test = test1 and test2
        return end_test
