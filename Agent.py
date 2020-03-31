import random
import numpy as np
from Qlearning import *

class Agent():
    def __init__(self, environnement, memory ,  choiceMethod ,  params, name = 'toto_01' ): #memory is an hyper parameter.
        self.environnement = environnement
        self.N_recommended = environnement.recommendation.n_recommended
        self.memory = memory
        self.state = []   #In this stack we remember the #memory last choices of the user
        self.previousState = []
        self.reward = 0 #reward at each step
        self.totalReward = 0#Total reward after an episode
        self.init_State()
        self.choiceMethod = choiceMethod #This defines how the Agent promotes new recommendations at each time
        self.recommendation = []
        self.name = name
        if self.choiceMethod == 'Qlearning':
            QLchoiceMethod = params['QLchoiceMethod']
            epsilon = params['epsilon']
            learning_rate = params['learning_rate']
            gamma = params['gamma']
            self.Qlearning = Qlearning(self, self.environnement.items.n_items, memory, self.N_recommended,epsilon  ,learning_rate , gamma, QLchoiceMethod )


    def init_State(self): #In order to still be able to make a recommendation at the begining (when the customer made no choice yet)
        for i in range(self.memory):
            self.state.append(-1) #Means that nothing was chosen at the beginning

    def updateStateAndReward(self, reward): #will help update the state at each step
        #keep previous values in case we want to train
        self.previousState = np.copy(self.state)

        self.state[:-1] = self.state[1:]
        self.state[-1] = self.environnement.customer.choice_id
        self.reward = reward
        self.totalReward += reward

    def recommend(self):#
        if self.choiceMethod == "random" :
            self.recommendation = np.random.choice(self.environnement.items.ids[:self.environnement.customer.choice_id]+self.environnement.items.ids[self.environnement.customer.choice_id+1:],self.N_recommended, replace= False)
        elif self.choiceMethod == "Qlearning":
            self.Qlearning.chooseAction()
            self.recommendation = self.Qlearning.recommendation
        else :
            print('Error : choiceMethod not recognized')

    def train(self):
        if self.choiceMethod == 'Qlearning':
            self.Qlearning.train()

    def endEpisode(self):
        self.totalReward = 0
        self.reward = 0
        self.state = []  # In this stack we remember the #memory last choices of the user
        self.previousState = []
        self.init_State()
        self.recommendation = []
        if self.choiceMethod == 'Qlearning':
            self.Qlearning.endEpisode()

    def display(self):
        print("------ AGENT DISPLAY ------")
        #print("*** Previous State ***")
        #print(self.previousState)
        print("*** Current State ***")
        print(self.state)
        print("*** Recommendation ***")
        print(self.recommendation)
        print("Current reward: "+str(self.reward))
        print("Total reward: "+ str(self.totalReward))
