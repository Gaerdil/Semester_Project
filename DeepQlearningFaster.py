#In this file, we implement the Qlearning method, but using function approximation.
#This is a neural network. The trainable parameters are the bias and weight tensor of each "trainable" layer
# state - action values .
# off-policy TD control
#Mathematical understanding/formula used : we can have a look at https://towardsdatascience.com/function-approximation-in-reinforcement-learning-85a4864d566

import random
import numpy as np
import torch

#HERE, WE WILL MAKE SOME HUGE APPROXIMATIONS OF MAX VALUE TO GET LESS TIME CONSUMING FOR LOOP

#Same thing than the Qlearning class, but with more complex actions.
#Indeed, before, one action was one item, now an action is a tuple (of all the recommended items).
class DeepQlearningFaster():
    #items_size is the number of items, memory the "memory" hyperparameter to define the states.
    def __init__(self, agent,  N_items, Memory, N_recommended, subset_size, epsilon = 0.1 ,learning_rate = 0.7, gamma = 0.3, choiceMethod = "eGreedy", debug = False):
        self.n_recommended = N_recommended
        self.memory = Memory
        self.n_items = N_items
        self.n_inputs = self.memory + 2*self.n_recommended
        self.agent = agent

        # --- TODO : remove after debug, so that all possible actions are not computed for very very large catalogs
        #Indeed, this part is only useful for debug and visualisation, but not essential
        self.debug = debug
        if self.debug:
            self.actions, self.actions_ids = self.initActions()
            self.numActions = len(self.actions_ids)
        elif not self.debug:
            self.numActions = 1 #This will instead allow us to see the number of action taken (when not in debug mode)
       # print(self.numActions)
        # ---


        self.lr = learning_rate
        self.choiceMethod = choiceMethod
        self.epsilon = epsilon
        self.gamma = gamma
        self.recommendation =  [] #will be updated  (the id of the choice)

        #The predefined subset of actions to make the for loops faster:
        self.subset_size = subset_size #TODO : change later if issues
        self.subset = []

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
        #TODO : remove line below later - just for debug, when all possible actions are computed
        if self.debug:
            self.recommendation_id = self.actions.index(self.recommendation)
        elif not self.debug:
            self.recommendation_id = 0

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



#Helper function to pick the predefined subset of actions in the action space. No need for computing in advance the actions!
    def pickActionsSubset(self, current_item): #create a predifined subset of possible actions
        self.subset = [] #No action in the subset

        #Case 1 : there are exactly or less than n_recommended cached items (the current item can be one of them):
        if self.agent.environnement.items.number_of_cached_items <= self.n_recommended :
            action = [i for i in self.agent.environnement.items.cached_items_ids if i != current_item]
            action_2ndPart = np.random.choice(self.agent.environnement.items.ids,self.n_recommended - len(action), replace=False)
            while current_item in action_2ndPart :
                action_2ndPart = np.random.choice(self.agent.environnement.items.ids, self.n_recommended - len(action),replace=False)
            action = action + list(action_2ndPart)
            self.subset.append(action[:])
            while len(self.subset) < self.subset_size:
                action = np.random.choice(self.agent.environnement.items.ids,self.n_recommended,replace=False)#replece : never two times the same item in same recommendation
                while current_item in action: #/!\ not propose the currently watched item...
                    action = np.random.choice(self.agent.environnement.items.ids,self.n_recommended, replace=False)  # replece : never two times the same item in same recommendation
                self.subset.append(list(action[:]))


        #Case 2 : we have enough items in the cached items (We might end up with repetitions though)
        else:
            while len(self.subset) < self.subset_size:
                action = np.random.choice(self.agent.environnement.items.cached_items_ids,self.n_recommended,replace=False)#replece : never two times the same item in same recommendation
                while current_item in action: #/!\ not propose the currently watched item...
                    action = np.random.choice(self.agent.environnement.items.cached_items_ids,self.n_recommended, replace=False)  # replece : never two times the same item in same recommendation
                self.subset.append(list(action[:]))
       # print(self.subset, current_item)

    #---- THE ONLY PLACE WHERE DeepQlearning and DeepQlearningFaster are different-----
    # We will speed up the "for loop process". Indeed, we are not that much looking for the absolute best action, but rather for
    #a not too bad action: an action that satisfies enough the customer, but meanwhile has a low caching cost.
    #  This is why, at each "chooseMaxAction" step, we will only choose the maximum action out of a predefined subset of actions.
    #This predefined subset of action will be chosen randomly at each time (each call to chooseMaxAction)
    #This will make the learning process less precise, as the TD target will also be estimated within the predifined subset of actions.
    #Hopefully, this should still converge to an acceptable solution - in a more reasonable time-, instead of the perfect solution at each time,out of billions of items.

    def chooseMaxAction(self, state):
        #This function could be improved in order to be "parallelized", or more efficient.
        #However, this work is mainly for theoritical study, we are letting aside the
        # 'efficiency' aspect of the code for the moment
        current_item = self.agent.environnement.customer.choice_id
        best_indice = None
        best_value = -np.inf
        self.pickActionsSubset(current_item) #actualise a plausible subset of items
        #for i in self.actions_ids: #TODO : change this line!
        #    action = self.actions[i][:]  #TODO : change this line!
        for i,action in enumerate(self.subset): #The for loop is only through the subset!
            value = self.getValue(state,action)
            if value > best_value:
                best_indice = i
                best_value = value
        #return  self.actions[best_indice][:] , best_value  #TODO : change this line!
        return self.subset[best_indice][:], best_value
    #----------------------------------------------------------------------------------


    def getValue(self,state,action):
        input = np.array(self.getInput(state, action))#/self.n_items  #Normalization step : to change in order to get something consistent with when the dataset will be updated
        input_tensor = torch.from_numpy(input).float()
        output_value = self.model(input_tensor)
        #print(output_value, state, action)
        return output_value.tolist()[0]

        # Here we "transform" the input in order to get costs and similarities, instead of item id, as an input
        # TODO : see why the basic item_id inputs was not working at all
    def getInput(self, state, action):
        input = []
        # Adding costs of states in memory:
        for item_id in state:
            input.append(self.agent.environnement.items.items[item_id].cost)
        # Adding costs of items of the action:
        for item_id in action:
            input.append(self.agent.environnement.items.items[item_id].cost)
        # Adding similarities of the last state with all items in action
        for item_id in action:
            input.append(self.agent.environnement.items.similarities[state[-1]][item_id])
        # print(state,action,  input)
        return input


    def train(self):

        #The TD error
        current_state_value = self.chooseMaxAction(self.agent.state)[1]
        last_state_value = self.getValue(list(self.agent.previousState), self.recommendation)
        delta = self.agent.reward + self.gamma * current_state_value - last_state_value

        #Computing the gradients -----------------------------------
        #grads = self.computeGrads(list(self.agent.previousState),self.recommendation, last_state_value)
        #grads = self.computeGrads(last_state_value)
        #grads = self.computeGrads(list(self.agent.previousState), self.recommendation)
        X = np.array(self.getInput(list(self.agent.previousState), self.recommendation))# / self.n_items  # Normalization step : to change in order to get something consistent with when the dataset will be updated
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



    def value_loss(self,X): #This is the "loss function" used by torch to backpropagate (as X will be the score)
        return X

    def endEpisode(self):
        self.recommendation = []

    def display(self, print_weights_bias = True,print_actions = False): #/!\ print actions only if debug mode = True
        print("--------------------------> Deep Q learning ( smaller for loops ) method :")
        print(" memory: " + str(self.memory))
        print(" number of items to recommend at each step : " + str(self.n_recommended))
        print('Subset size: '+str(self.subset_size))
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

    # def setDebugMode(self): #Helper function if we want to change the debug mode
    #     self.debug = True
    #     self.actions, self.actions_ids = self.initActions()
    #     self.numActions = len(self.actions_ids)
    #     self.agent.choicesThisEpisode = np.zeros(self.numActions)