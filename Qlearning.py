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
        self.initQtable()
        self.lr = learning_rate
        self.choiceMethod = choiceMethod
        self.epsilon = epsilon
        self.gamma = gamma
        self.recommendation =  [] #will be updated  (the id of the choice)
        self.N_recommended = N_recommended
        self.agent = agent

    def initQtable(self): #We have to initialize the Qtable (and remove the possibility of recommending the currently watched item)
        #self.Qtable = np.ones(self.dims)  # not flattened Qtable. Optimistic values (ones) to encourage exploration at the beginning.
        #self.Qtable = np.ones(self.dims)
        # Or we can also set everything to 0
        self.Qtable = np.zeros(self.dims)
        # For instance, Qtable[0][5][3] would allow us to get the list of values for actions, for the state [0,5,3] (if memory = 3)

    def chooseAction(self): #When we are in a new state, we get to choose a new action
        if self.choiceMethod == "eGreedy" :
            self.chooseActionEGreedy()
        else:
            print("Error : Qlearning Choice Method not recognized")

    def chooseActionEGreedy(self):
        self.recommendation = []
        rand_ = random.random()
        if rand_ <= self.epsilon:
            self.chooseActionRandomLast()
        else:
            self.recommendation = self.chooseMaxAction(self.agent.state, self.N_recommended)

    def chooseActionRandom(self):
        self.recommendation = np.random.choice(self.agent.environnement.items.ids[:self.agent.environnement.customer.choice_id]+self.agent.environnement.items.ids[self.agent.environnement.customer.choice_id+1:],self.N_recommended, replace= False)

    def chooseActionRandomLast(self):
        maxChoices = self.chooseMaxAction(self.agent.state, self.N_recommended - 1)
        randomChoice = np.random.choice(self.agent.environnement.items.ids[:self.agent.environnement.customer.choice_id]+self.agent.environnement.items.ids[self.agent.environnement.customer.choice_id + 1:],1, replace=False)
        while randomChoice in maxChoices :
            randomChoice = np.random.choice(self.agent.environnement.items.ids[:self.agent.environnement.customer.choice_id] + self.agent.environnement.items.ids[self.agent.environnement.customer.choice_id + 1:],1, replace=False)
        self.recommendation = maxChoices[:].tolist()
        self.recommendation.append(randomChoice[0])


    #(We need to update this function to add randomness! (not select the n_recommended higher in the order of their id...)
    def chooseMaxAction(self, state, num):
        self.Qtable[tuple(state)][state[-1]] = -np.inf  #to make sure we wont recommend the currently watched item
        #return (np.flip(self.Qtable[tuple(state)]).argsort()[- num :])
        return self.selectRandomlyMax(np.copy(self.Qtable[tuple(state)][:]), num, [])

    def train(self, print_ = False):
        if print_:
            self.display()

        prev_state_reco = self.Qtable[tuple(self.agent.previousState)][self.agent.environnement.customer.choice_id]
        current_state_max = self.Qtable[tuple(self.agent.state)][self.chooseMaxAction(self.agent.state, 1)][0]

        self.Qtable[tuple(self.agent.previousState)][self.agent.environnement.customer.choice_id] = prev_state_reco + self.lr * (self.agent.reward + self.gamma * current_state_max - prev_state_reco)

    def endEpisode(self):
        self.recommendation = []

    def display(self):
        print("--------------------------> Q learning method :")
        print(" learning rate: "+str(self.lr))
        print(" gamma: "+str(self.gamma))
        print(">>>>> Qtable <<<<<")
        print(self.Qtable)


#Helper functions :


    def selectRandomlyMax(self, li, n, results): #select randomly n maximums in an np.array list. Returns indices.
        if n == 0:
            return np.array(results)
        else :
            max_ = np.max(li)
            ind = np.argwhere(li == max_).flatten()
            choice = np.random.choice(ind)
            results.append(choice)
            li[choice] = -np.inf
            return self.selectRandomlyMax(li, n-1, results)
