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

        self.choicesLastSerie_total = np.zeros(environnement.items.n_items)


        if display_avg:
            startTime = time.time()
            print("------------------> Average of series begins:  <------------------")
            print(str(num_avg)+" independent training/testing processes")
            print("environnement name: "+environnement.name)
            print("Memory size: " + str(memory))
            print("Number of items to recommend: "+ str(environnement.recommendation.n_recommended))
            print("--- We will test the following hyperparameters ---")
            print("choice method: " + choiceMethod)
            print("epochs: "+ str(epochs))
            print("Reward hyper parameters: "+ str(environnement.rewardParameters))
            if choiceMethod == 'Qlearning' or choiceMethod == 'QlearningActionsTuples' :
                print(params)

        agent = Agent(environnement, memory, choiceMethod, params)
        series = Series(environnement, agent, epochs, train_list, steps, display, displayItems)

        self.avgRewards = np.array(series.allRewards[:])
        if choiceMethod == "QlearningActionsTuples":
            self.choicesLastSerieActionTuples_total = np.zeros(agent.Qlearning.numActions)

        for a in tqdm(range(num_avg)):
            # We keep the exact same environnement, but reinitialize the Q-table (testing if we were just lucky in the learning process)
            agent = Agent(environnement, memory, choiceMethod, params)
            series = Series(environnement, agent, epochs, train_list, steps,  display, displayItems)
            self.avgRewards =  self.avgRewards  +  np.array(series.allRewards[:])
            self.choicesLastSerie_total= self.choicesLastSerie_total + series.choicesLastSerie
            if choiceMethod == "QlearningActionsTuples":
                self.choicesLastSerieActionTuples_total = self.choicesLastSerieActionTuples_total + series.choiceslastSerieActionTuples

        self.avgRewards = self.avgRewards/num_avg
        self.avgLastReward = self.avgRewards[-1]


        if display_avg:
            endTime = time.time()
            print(" \n \n Execution time: "+str(endTime - startTime))

            print("Qtable of the last series ------------------------------>")
            print(agent.Qlearning.Qtable)
            print("---------------------------------------------------->")
            print(" ")
            print("After the learning process : how often  is an item recommended? (total of all series) ")
            print(self.choicesLastSerie_total)
            if choiceMethod == "QlearningActionsTuples":
                print("After the learning process : how often  is an Action tuple recommended? (total of all series) ")
                print("Action list:")
                print(agent.Qlearning.actions)
                print("Action ids list:")
                print(agent.Qlearning.actions_ids)
                print("Number of time selected (per action id):")
                print(self.choicesLastSerieActionTuples_total)

            print("------------------> Series ends <------------------")

class Series(): #several series, to show the whole learning/testing process
    def __init__(self, environnement, agent, epochs, train_list, steps=5,  display=True, displayItems=False):

        if display:
            print("------------------> Series begins <------------------")

        self.allRewards = []
        for train_ in train_list:
            serie = Serie(environnement, agent, epochs, steps, train_, display, displayItems)
           # self.allRewards = self.allRewards + serie.serieRewards[:]
            self.allRewards.append(np.mean(serie.serieRewards[:]))
        self.choicesLastSerie = serie.choicesThisSerie[:] #Equal to the choices done at the last "not training" serie (end of the learning process)
        if agent.choiceMethod == "QlearningActionsTuples":
            self.choiceslastSerieActionTuples = serie.choicesThisSerieActionTuples

        if display:
            print("------------------> Series ends <------------------")




class Serie(): #a serie is a serie of episodes with all the same train_ type (true or false).
    # train_ indicates if the agent is going to updates its Qtable during the episode (training)
    def __init__(self, environnement, agent, epochs, steps = 5, train_ = False, display = True , displayItems  = False):
        self.serieRewards = []
        self.choicesThisSerie = np.zeros(environnement.items.n_items)

        if display :
            self.display(train_)

        if agent.choiceMethod == "QlearningActionsTuples":
            self.choicesThisSerieActionTuples = np.zeros(agent.Qlearning.numActions)

        for epoch in range(epochs ):
            episode = Episode(environnement, agent, steps, train_, display, displayItems)
            self.serieRewards.append(np.mean(episode.episodeReward)) #Taking the mean should help get more consistent results
            self.choicesThisSerie = self.choicesThisSerie + episode.choicesThisEpisode
            if agent.choiceMethod == "QlearningActionsTuples":
                self.choicesThisSerieActionTuples = self.choicesThisSerieActionTuples + episode.choicesThisEpisodeActionTuples

    def display(self, train_):
        print("------------------> Serie begins")
        print("Training session: "+str(train_))