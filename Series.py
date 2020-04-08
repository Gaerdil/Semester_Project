# A serie is composed of several episodes

import random
import numpy as np
from Episode import *
import time
from tqdm import tqdm



from Agent import *


class AverageSeries(): #Helpful to get a better unbiased statistical estimate of the efficiency of our model
    #We redo the learning process several times and average the results to reduce the amount of randomness in our results

    def __init__(self, num_avg, environnement, memory, choiceMethod, params, epochs, train_list, steps=5,display_avg = True, display=True, displayItems=False):


        if display_avg:
            startTime = time.time()
            print("------------------> Average of series begins:  <------------------")
            print(str(num_avg)+" independent training/testing processes")
            print("environnement name: "+environnement.name)
            print("--- We will test the following hyperparameters ---")
            print("choice method: " + choiceMethod)
            print("epochs: "+ str(epochs))
            print("Reward hyper parameters: "+ str(environnement.rewardParameters))
            if choiceMethod == 'Qlearning':
                print(params)

        agent = Agent(environnement, memory, choiceMethod, params)
        series = Series(environnement, agent, epochs, train_list, steps, display, displayItems)

        self.avgRewards = np.array(series.allRewards[:])

        for a in tqdm(range(num_avg)):
            # We keep the exact same environnement, but reinitialize the Q-table (testing if we were just lucky in the learning process)
            agent = Agent(environnement, memory, choiceMethod, params)
            series = Series(environnement, agent, epochs, train_list, steps,  display, displayItems)
            self.avgRewards =  self.avgRewards  +  np.array(series.allRewards[:])
        self.avgRewards = self.avgRewards/num_avg

        if display_avg:
            endTime = time.time()
            print(" \n \n Execution time: "+str(endTime - startTime))
            print("------------------> Series ends <------------------")

        print("Qtable of the last series ------------------------------>")
        print(agent.Qlearning.Qtable)
        print("---------------------------------------------------->")


class Series(): #several series, to show the whole learning/testing process
    def __init__(self, environnement, agent, epochs, train_list, steps=5,  display=True, displayItems=False):

        if display:
            print("------------------> Series begins <------------------")

        self.allRewards = []
        for train_ in train_list:
            serie = Serie(environnement, agent, epochs, steps, train_, display, displayItems)
           # self.allRewards = self.allRewards + serie.serieRewards[:]
            self.allRewards.append(np.mean(serie.serieRewards[:]))


        if display:
            print("------------------> Series ends <------------------")




class Serie(): #a serie is a serie of episodes with all the same train_ type (true or false).
    # train_ indicates if the agent is going to updates its Qtable during the episode (training)
    def __init__(self, environnement, agent, epochs, steps = 5, train_ = False, display = True , displayItems  = False):
        self.serieRewards = []

        if display :
            self.display(train_)

        for epoch in range(epochs):
            episode = Episode(environnement, agent, steps, train_, display, displayItems)
            self.serieRewards.append(np.mean(episode.episodeReward)) #Taking the mean should help get more consistent results

    def display(self, train_):
        print("------------------> Serie begins")
        print("Training session: "+str(train_))