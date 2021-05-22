# Import libraries
from IPython.display import clear_output
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class GameScore:

    def __init__(self, window_len=100,frequency_display = 10,display_chart=True,clear_output=True):
        self.window_len = window_len
        self.frequency_display = frequency_display
        self.display_chart = display_chart
        self.clear_output = clear_output

        # Initialization
        self.episod_index = 0
        # list containing scores from each episode
        self.all_scores = []
        self.all_moving_averages=[]
        # last [window_len] scores - useful for calculating a moving average
        self.scores_window = deque(maxlen=window_len)
        self.last_score = 0
        self.last_moving_average  = 0

    def update_scores(self,episod_score):

        self.episod_index+=1
        self.all_scores.append(episod_score)
        self.scores_window.append(episod_score)
        self.all_moving_averages.append(np.mean(self.scores_window))

        # Update general statistics
        self.last_score = self.all_scores[-1]
        self.last_moving_average  = self.all_moving_averages[-1]

    def display_score(self):

        if self.episod_index % self.frequency_display == 0:
            clear_output(wait=self.clear_output)
            print ("Last Score: {:.2f} - Moving Average over last {} episods: {:.2f}".\
                   format(self.last_score, self.episod_index, self.last_moving_average))
            if self.display_chart:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.plot(np.arange(++len(self.all_scores)), self.all_scores)
                plt.ylabel('Score')
                plt.xlabel('Episode #')


                plt.show()
