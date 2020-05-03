import random
import numpy as np
from Qlearning import *
from QlearningActionTuples import *
from LinearQlearning import *
from PolynomialQlearning import *
from SimpleDeepQlearning import *

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


        elif self.choiceMethod == 'QlearningActionsTuples':
            QLchoiceMethod = params['QLchoiceMethod']
            epsilon = params['epsilon']
            learning_rate = params['learning_rate']
            gamma = params['gamma']
            self.Qlearning = QlearningActionTuples(self, self.environnement.items.n_items, memory, self.N_recommended,epsilon  ,learning_rate , gamma, QLchoiceMethod )
            self.choicesThisEpisode = np.zeros(self.Qlearning.numActions)

        elif self.choiceMethod == 'LinearQlearning':
            QLchoiceMethod = params['QLchoiceMethod']
            epsilon = params['epsilon']
            learning_rate = params['learning_rate']
            gamma = params['gamma']
            self.Qlearning = LinearQlearning(self, self.environnement.items.n_items, memory, self.N_recommended,
                                                   epsilon, learning_rate, gamma, QLchoiceMethod)
            self.choicesThisEpisode = np.zeros(self.Qlearning.numActions)

        elif self.choiceMethod == 'PolynomialQlearning':
            QLchoiceMethod = params['QLchoiceMethod']
            epsilon = params['epsilon']
            learning_rate = params['learning_rate']
            gamma = params['gamma']
            degree = params['degree']
            self.Qlearning = PolynomialQlearning(self, degree, self.environnement.items.n_items, memory, self.N_recommended,
                                             epsilon, learning_rate, gamma, QLchoiceMethod)
            self.choicesThisEpisode = np.zeros(self.Qlearning.numActions)

        elif self.choiceMethod == 'SimpleDeepQlearning':
            QLchoiceMethod = params['QLchoiceMethod']
            epsilon = params['epsilon']
            learning_rate = params['learning_rate']
            gamma = params['gamma']
            hidden_size = params['hidden_size']
            self.Qlearning = SimpleDeepQlearning(self, hidden_size, self.environnement.items.n_items, memory,
                                                 self.N_recommended,
                                                 epsilon, learning_rate, gamma, QLchoiceMethod)
            self.choicesThisEpisode = np.zeros(self.Qlearning.numActions)



    def init_State(self): #In order to still be able to make a recommendation at the begining (when the customer made no choice yet)
        li = [self.environnement.customer.previous_choice_id, self.environnement.customer.choice_id]
        for i in range(self.memory-2):
           # self.state.append(-1) #Means that nothing was chosen at the beginning old false version
           random_ = random.randint(0, self.environnement.items.n_items-1)
           while random_ == li[0]:
               random_ = random.randint(0, self.environnement.items.n_items-1)
           li = [random_] + li

        if self.memory == 1 :
            self.state = [self.environnement.customer.choice_id]
        else:
            self.state = li

    def updateStateAndReward(self, reward): #will help update the state at each step
        #keep previous values in case we want to train
        self.previousState = np.copy(self.state)

        self.state[:-1] = self.state[1:]
        self.state[-1] = self.environnement.customer.choice_id
        self.reward = reward
        self.totalReward += reward

    def recommend(self, train_=False):#
        if self.choiceMethod == "random" :
            self.recommendation = np.random.choice(self.environnement.items.ids[:self.environnement.customer.choice_id]+self.environnement.items.ids[self.environnement.customer.choice_id+1:],self.N_recommended, replace= False)
        elif self.choiceMethod == "Qlearning"  :
            self.Qlearning.chooseAction()
            self.recommendation = self.Qlearning.recommendation
        elif self.choiceMethod in ["QlearningActionsTuples", "LinearQlearning","PolynomialQlearning", "SimpleDeepQlearning"]:
            self.Qlearning.chooseAction(train_)
            self.recommendation = self.Qlearning.recommendation
            self.choicesThisEpisode[self.Qlearning.recommendation_id] += 1
        else :
            print('Error : choiceMethod not recognized in agent.recommend() function')

    def train(self):
        if self.choiceMethod != 'random' :
            self.Qlearning.train()

    def endEpisode(self):
        self.totalReward = 0
        self.reward = 0
        self.state = []  # In this stack we remember the #memory last choices of the user
        self.previousState = []
        self.init_State()
        self.recommendation = []
        if self.choiceMethod != "random" and self.choiceMethod != "Qlearning":
            self.choicesThisEpisode = np.zeros(self.Qlearning.numActions)
            self.Qlearning.endEpisode()
        else:
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
