# A serie is composed of several episodes

import random
import numpy as np
from Episode import *


class Series(): #several series, to show the whole learning/testing process
    def __init__(self, environnement, agent, epochs, train_list, steps=5,  display=True, displayItems=False):

        if display:
            print("------------------> Series begins <------------------")

        self.allRewards = []
        for train_ in train_list:
            serie = Serie(environnement, agent, epochs, steps, train_, display, displayItems)
            self.allRewards = self.allRewards + serie.serieRewards[:]

        if display:
            print("------------------> Series ends <------------------")




class Serie(): #a serie is a serie of episodes with all the same train_ type (true or false)
    def __init__(self, environnement, agent, epochs, steps = 5, train_ = False, display = True , displayItems  = False):
        self.serieRewards = []

        if display :
            self.display(train_)

        for epoch in range(epochs):
            episode = Episode(environnement, agent, steps, train_, display, displayItems)
            self.serieRewards.append(episode.episodeReward)

    def display(self, train_):
        print("------------------> Serie begins")
        print("Training session: "+str(train_))