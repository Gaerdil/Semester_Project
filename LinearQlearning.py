#In this file, we implement the Qlearning method, but using function approximation.
#We will begin by a very simple function approximation, a linear regression to estimate
# state - action values .
# off-policy TD control
#Mathematical understanding/formula used : we can have a look at https://towardsdatascience.com/function-approximation-in-reinforcement-learning-85a4864d566

import random
import numpy as np

#Same thing than the Qlearning class, but with more complex actions.
#Indeed, before, one action was one item, now an action is a tuple (of all the recommended items).
class LinearQlearning():
    #items_size is the number of items, memory the "memory" hyperparameter to define the states.
    def __init__(self, agent, N_items, Memory, N_recommended, epsilon = 0.1 ,learning_rate = 0.7, gamma = 0.3, choiceMethod = "eGreedy"):
        self.n_recommended = N_recommended
        self.memory = Memory
        self.n_items = N_items
        self.weights = np.random.rand(self.memory + self.n_recommended,1)  #The weights that will be used to estimate the state value
        self.n_inputs = len(self.weights)
        self.agent = agent
        self.actions, self.actions_ids = self.initActions()
        self.numActions = len(self.actions_ids)
        self.lr = learning_rate
        self.choiceMethod = choiceMethod
        self.epsilon = epsilon
        self.gamma = gamma
        self.recommendation =  [] #will be updated  (the id of the choice)


          # For instance, Qtable[0][5][3] would allow us to get the list of values for actions, for the state [0,5,3] (if memory = 3)


    def chooseAction(self, train_): #When we are in a new state, we get to choose a new action
        if train_ : #a training session : we choose the choice method specific to training sessions
            if self.choiceMethod == "eGreedy" :
                self.chooseActionEGreedy()
            else:
                print("Error : Qlearning Choice Method not recognized")
        else: #not a training session
            self.recommendation = self.chooseMaxAction(self.agent.state)[0]

    def chooseActionEGreedy(self):
        self.recommendation = []
        rand_ = random.random()
        if rand_ <= self.epsilon:
            self.recommendation = self.chooseMaxAction(self.agent.state)[0]
            self.chooseActionRandom()
        else:
            self.recommendation = self.chooseMaxAction(self.agent.state)[0]



    def chooseActionRandom(self):
        #We already have chosen the recommendation, according to the "chooseMaxAction" function.
        # But one of the items is going to be changed, so that it can be chosen randomly.
        random_recommendation = np.random.randint(0, self.n_items)
        while self.agent.environnement.customer.choice_id == random_recommendation or random_recommendation in self.recommendation:
            random_recommendation = np.random.randint(0, self.n_items)
        index = np.random.randint(0,self.n_recommended)
        self.recommendation[index] = random_recommendation


    def initActions(self): #helper function to compute the actions possibilities
        def computeActions(n,li):
            if n==1 :
                return li
            else :
                li = [prev+ [j] for prev in li for j in self.agent.environnement.items.ids if j not in prev ]
                return computeActions(n-1, li)

        actions = computeActions(self.n_recommended,[[i] for i in self.agent.environnement.items.ids] )
        return (actions,[i for i in range(len(actions))])


    def chooseMaxAction(self, state):
        #This function could be improved in order to be "parallelized", or more efficient.
        #However, this work is mainly for theoritical study, we are letting aside the
        # 'efficiency' aspect of the code for the moment
        current_item = self.agent.environnement.customer.choice_id
        best_indice = None
        best_value = -np.inf
        for i in self.actions_ids:
            action = self.actions[i][:]
            if current_item not in action:
                value = self.getValue(state,action)
                if value > best_value:
                    best_indice = i
                    best_value = value
        return  self.actions[best_indice][:] , best_value

    # In the logistic-inspired model, this is just the matrix multiplication.
    def getValue(self,state,action):
        return np.dot(np.array(state+action), self.weights)[0]


    def train(self, print_ = False):   #/!\ to update
        if print_:
            self.display()

        #The TD error
        delta = self.agent.reward  + self.chooseMaxAction(self.agent.state)[1] - self.getValue(list(self.agent.previousState), self.recommendation)

        #Finally, the gradient descent update
        self.weights = self.weights + self.lr*delta*np.array(list(self.agent.previousState) + self.recommendation).reshape(self.n_inputs,1)




        #prev_state_reco = self.Qtable[tuple(self.agent.previousState)][self.recommendation_id]
        #current_state_max = self.Qtable[tuple(self.agent.state)][self.chooseMaxAction(self.agent.state, 1)][0]
        #self.Qtable[tuple(self.agent.previousState)][self.recommendation_id] = prev_state_reco + self.lr * (self.agent.reward + self.gamma * current_state_max - prev_state_reco)



    def endEpisode(self):
        self.recommendation = []

    def display(self, print_actions = False):
        print("--------------------------> Q learning (simple Linear approximation) method :")
        print(" memory: " + str(self.memory))
        print(" number of items to recommend at each step : " + str(self.n_recommended))
        print(" learning rate: "+str(self.lr))
        print(" gamma: "+str(self.gamma))
        print("Weights: ")
        print(self.weights)
        if print_actions:
            print(self.actions)

#Helper functions