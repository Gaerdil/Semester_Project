#In this file, we implement the Qlearning method, but using function approximation.
#This is a neural network. The trainable parameters are the bias and weight tensor of each "trainable" layer
# state - action values .
# off-policy TD control
#Mathematical understanding/formula used : we can have a look at https://towardsdatascience.com/function-approximation-in-reinforcement-learning-85a4864d566

import random
import numpy as np
import torch
from torch import nn
import copy


#Same thing than the Qlearning class, but with more complex actions.
#Indeed, before, one action was one item, now an action is a tuple (of all the recommended items).
class DeepQlearning():
    #items_size is the number of items, memory the "memory" hyperparameter to define the states.
    def __init__(self, agent,  N_items, Memory, N_recommended, epsilon = 0.1 ,learning_rate = 0.7, gamma = 0.3, choiceMethod = "eGreedy"):
        self.n_recommended = N_recommended
        self.memory = Memory
        self.n_items = N_items
        self.n_inputs = self.memory + self.n_recommended
        self.agent = agent
        self.actions, self.actions_ids = self.initActions()
        self.numActions = len(self.actions_ids)
        self.lr = learning_rate
        self.choiceMethod = choiceMethod
        self.epsilon = epsilon
        self.gamma = gamma
        self.recommendation =  [] #will be updated  (the id of the choice)


    def setModel(self, Model,Trainable_layers): #A separate step, so that we can have the input size from the Qlearning __init__
        self.model = Model
        self.trainable_layers = Trainable_layers
          # For instance, Qtable[0][5][3] would allow us to get the list of values for actions, for the state [0,5,3] (if memory = 3)


    def chooseAction(self, train_): #When we are in a new state, we get to choose a new action
        if train_ : #a training session : we choose the choice method specific to training sessions
            if self.choiceMethod == "eGreedy" :
                self.chooseActionEGreedy()
            else:
                print("Error : Qlearning Choice Method not recognized")
        else: #not a training session
            self.recommendation = self.chooseMaxAction(self.agent.state)[0]
        self.recommendation_id = self.actions.index(self.recommendation)

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


    def getValue(self,state,action):
        input = np.array(state+action)#/self.n_items  #Normalization step : to change in order to get something consistent with when the dataset will be updated
        input_tensor = torch.from_numpy(input).float()
        output_value = self.model(input_tensor)
        #print(output_value, state)
        return output_value.tolist()[0]


    def train(self):

        #The TD error
        current_state_value = self.chooseMaxAction(self.agent.state)[1]
        last_state_value = self.getValue(list(self.agent.previousState), self.recommendation)
        delta = self.agent.reward + self.gamma * current_state_value - last_state_value

        #Computing the gradients -----------------------------------
        #grads = self.computeGrads(list(self.agent.previousState),self.recommendation, last_state_value)
        #grads = self.computeGrads(last_state_value)
        #grads = self.computeGrads(list(self.agent.previousState), self.recommendation)
        X = np.array(list(self.agent.previousState)+ self.recommendation)# / self.n_items  # Normalization step : to change in order to get something consistent with when the dataset will be updated
        X_tensor = torch.from_numpy(X).float()
        y_pred = self.model(X_tensor)
        loss = self.value_loss(y_pred)
        loss.backward()
        #nn.utils.clip_grad_value_(self.model.parameters(), 5)
        #Gradient computed -----------------------------------------

        # Finally, the gradient descent update
        for i in self.trainable_layers:
            #Clip gradients
            self.model[i].weight.grad.data.clamp_(-0.1,0.1)
            self.model[i].bias.grad.data.clamp_(-0.1, 0.1)

           # print("LAyer "+str(i))
           # print("Before")
           # print(self.model[i].weight.data)
           # print(self.model[i].bias.data)

           # print("------------delta, lr & grads ---------------")
           # print("delta: "+str(delta)+" , lr: "+str(self.lr))
           # print("GRADIENTS:")
           # print(self.model[i].weight.grad.data)
           # print(self.model[i].bias.grad.data)
           # print(" >>>>>>>>> UPDATE AMOUNT <<<<<<<<<<<")
           # print(self.lr * delta * self.model[i].weight.grad.data)
           # print(self.lr * delta * self.model[i].bias.grad.data)
           # print("---------------------------")

            self.model[i].weight.data = self.model[i].weight.data + self.lr * delta * self.model[i].weight.grad.data
            self.model[i].bias.data = self.model[i].bias.data + self.lr * delta * self.model[i].bias.grad.data
            self.model[i].weight.grad.data.zero_()
            self.model[i].bias.grad.data.zero_()

            #print("After")
            #print(self.model[i].weight.data)
            #print(self.model[i].bias.data)

    # TODO : Remove computeGrads as torch already takes care of it
    #def computeGrads(self,prev_state, prev_action, last_state_value):
     #def computeGrads(self, last_state_value):
    def computeGrads(self, prev_state, prev_action):
        X = np.array(prev_state + prev_action) / self.n_items  # Normalization step : to change in order to get something consistent with when the dataset will be updated
        X_tensor = torch.from_numpy(X).float()
        y_pred = self.model(X_tensor)
        loss = self.value_loss(y_pred)
        loss.backward()

        grads = {}
        #for i in self.trainable_layers:

        return grads

    def value_loss(self,X): #This is the "loss function" used by torch to backpropagate (as X will be the score)
        return X

    def endEpisode(self):
        self.recommendation = []

    def display(self, print_weights_bias = True,print_actions = False):
        print("--------------------------> Q learning ( neural network approximation) method :")
        print(" memory: " + str(self.memory))
        print(" number of items to recommend at each step : " + str(self.n_recommended))
        print(" learning rate: "+str(self.lr))
        print(" gamma: "+str(self.gamma))
        print("trainable layers ids: "+str(self.trainable_layers))
        print("Model:")
        print(self.model)
        if print_weights_bias:
            for i in self.trainable_layers:
                print('\n <---- Weight and bias of Layer ' + str(i) + ' ---->')
                print("Weight -->")
                print(self.model[i].weight.data)
                print('Bias -->')
                print(self.model[i].bias.data)
        if print_actions:
            print(self.actions)
        print('----------------------------------------------------')

