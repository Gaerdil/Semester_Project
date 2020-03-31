#In this file, we implement the Qlearning method
# off-policy TD control

import random
import numpy as np

class Qlearning():
    #items_size is the number of items, memory the "memory" hyperparameter to define the states.
    def __init__(self, agent, items_size, memory, N_recommended,epsilon = 0.1 ,learning_rate = 0.7, gamma = 0.3, choiceMethod = "eGreedy"):
        #The number of states is here equal to items_size ** memory .
        self.numStates = items_size**memory
        self.dims =  [items_size for i in range(memory+ 1)] #the +1 for the actions
        self.Qtable = np.ones(self.dims) #not flattened Qtable. Optimistic values (ones) to encourage exploration at the beginning.
        #For instance, Qtable[0][5][3] would allow us to get the list of values for actions, for the state [0,5,3] (if memory = 3)
        self.lr = learning_rate
        self.choiceMethod = choiceMethod
        self.epsilon = epsilon
        self.gamma = gamma
        self.recommendation =  [] #will be updated  (the id of the choice)
        self.N_recommended = N_recommended
        self.agent = agent

    def chooseAction(self): #When we are in a new state, we get to choose a new action
        if self.choiceMethod == "eGreedy" :
            self.chooseActionEGreedy()
        else:
            print("Error : Qlearning Choice Method not recognized")

    def chooseActionEGreedy(self):
        rand_ = random.random()
        if rand_ <= self.epsilon:
            self.chooseActionRandom()
        else:
            self.recommendation = self.chooseMaxAction(self.agent.state)

    def chooseActionRandom(self):
        self.recommendation = np.random.choice(self.agent.environnement.items.ids[:self.agent.environnement.customer.choice_id]+self.agent.environnement.items.ids[self.agent.environnement.customer.choice_id+1:],self.N_recommended, replace= False)

    def chooseMaxAction(self, state):
        return (self.Qtable[tuple(state)]).argsort()[-self.N_recommended:]

    def train(self, print_ = False):
        #print(self.Qtable[tuple(self.agent.state)])
        #print(self.recommendation)
        #print(self.Qtable[tuple(self.agent.state)][self.recommendation])
        #print(self.Qtable[tuple(self.agent.previousState)])
        if print_:
            self.display()

        prev_state_reco = self.Qtable[tuple(self.agent.previousState)][self.recommendation]
        current_state_max = self.Qtable[tuple(self.agent.state)][self.chooseMaxAction(self.agent.state)]
        self.Qtable[tuple(self.agent.previousState)][self.recommendation] = prev_state_reco + self.lr*(self.agent.reward+self.gamma*current_state_max-prev_state_reco)

    def endEpisode(self):
        self.recommendation = []

    def display(self):
        print("--------------------------> Q learning method :")
        print(" learning rate: "+str(self.lr))
        print(" gamma: "+str(self.gamma))
        print(">>>>> Qtable <<<<<")
        print(self.Qtable)