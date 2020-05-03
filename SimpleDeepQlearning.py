#In this file, we implement the Qlearning method, but using function approximation.
#We will begin by a simple function approximation, a small neural network (one hidden layer) to estimate
# state - action values .
# off-policy TD control
#Mathematical understanding/formula used : we can have a look at https://towardsdatascience.com/function-approximation-in-reinforcement-learning-85a4864d566

import random
import numpy as np


#Same thing than the Qlearning class, but with more complex actions.
#Indeed, before, one action was one item, now an action is a tuple (of all the recommended items).
class SimpleDeepQlearning():
    #items_size is the number of items, memory the "memory" hyperparameter to define the states.
    def __init__(self, agent, Hidden_size, N_items, Memory, N_recommended, epsilon = 0.1 ,learning_rate = 0.7, gamma = 0.3, choiceMethod = "eGreedy"):
        self.n_recommended = N_recommended
        self.memory = Memory
        self.n_items = N_items
        self.hidden_size = Hidden_size
        self.n_inputs = self.memory + self.n_recommended
        self.weights1 = np.random.normal(0,np.sqrt(2/self.n_inputs),(self.n_inputs,self.hidden_size))  #/(self.n_inputs)  #The weights that will be used to estimate the state value
        self.weights2 = np.random.normal(0,np.sqrt(2/self.hidden_size),(self.hidden_size,1)) #/ self.hidden_size  #normalization
        self.bias1 = np.random.rand(self.hidden_size)
        self.bias2 = np.random.rand()
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
        normalized_input = np.array(state+action)/self.n_items  #Normalization step : to change in order to get something consistent with when the dataset will be updated
        hidden_layer = relu(np.dot(normalized_input, self.weights1) + self.bias1)
        #output_value = sigmoid(np.dot(hidden_layer, self.weights2) + self.bias2) #The sigmoid model turned out to be not working the expected way
        output_value = np.dot(hidden_layer, self.weights2) + self.bias2
        return output_value[0]


    def train(self, print_ = False):   #/!\ to update
        if print_:
            self.display()

        #The TD error
        current_state_value = self.chooseMaxAction(self.agent.state)[1]
        last_state_value = self.getValue(list(self.agent.previousState), self.recommendation)
        delta = self.agent.reward + self.gamma * current_state_value - last_state_value

        #Computing the gradients
        grads = self.computeGrads(list(self.agent.previousState),self.recommendation, last_state_value)

        # Finally, the gradient descent update
        #self.weights1 = self.weights1 + self.lr*delta*np.array(list(self.agent.previousState) + self.recommendation).reshape(self.n_inputs,1)
        self.weights1 = self.weights1 + self.lr * delta *grads['w1']
        self.weights2 = self.weights2 + self.lr * delta * grads['w2']
        self.bias1 = self.bias1 + self.lr * delta * grads['b1']
        self.bias2 = self.bias2 + self.lr * delta * grads['b2']

    def computeGrads(self,prev_state, prev_action, last_state_value):
        X = np.array(prev_state + prev_action)/self.n_items #Normalized input

        #Linear step of computing the hidden layer
        hidden_layer_linear = np.dot(X, self.weights1) + self.bias1 #Intermediary step and useful value. Could have been kept in memory for efficiency, when we had to choose an action.
        #non linear function activation
        hidden_layer_activation = relu(hidden_layer_linear).reshape((self.hidden_size,1))

        # bias2
        #b2 = last_state_value*(1-last_state_value)#sigmoid model
        b2 = 1 #non sigmoid model
        # weight2
        w2 = b2*hidden_layer_activation[:]

        # bias1
        #b1 = (last_state_value*(1-last_state_value)*self.weights2*((drelu(hidden_layer_linear)).reshape((self.hidden_size,1)))).reshape(self.hidden_size,)
        b1 = b2*( self.weights2 * ((drelu(hidden_layer_linear)).reshape((self.hidden_size, 1)))).reshape(self.hidden_size, )

        #weight1
        w1 = np.dot(X.reshape((self.n_inputs, 1)),b1.reshape((1,self.hidden_size)))

        return {'w1':w1,
                'w2': w2,
                'b1': b1,
                'b2':b2
                }

    def endEpisode(self):
        self.recommendation = []

    def display(self, print_actions = False):
        print("--------------------------> Q learning (simple neural network approximation) method :")
        print(" memory: " + str(self.memory))
        print(" number of items to recommend at each step : " + str(self.n_recommended))
        print(" learning rate: "+str(self.lr))
        print(" gamma: "+str(self.gamma))
        print("Hidden layer size: "+str(self.hidden_size))
        print("Hidden Weights: ")
        print(self.weights1)
        print("Output Weights: ")
        print(self.weights2)
        if print_actions:
            print(self.actions)

#Helper functions
def relu(X):
    return np.maximum(X, 0)

def sigmoid(X):
    return 1/(1+np.exp(-X))

def drelu(X):
    x = X[:]
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def dsigmoid(X):
    return sigmoid(X)*(1-sigmoid(X))