import random
from Items import *
from Recommendation import *
from Customer import *

class Environnement():
    def __init__(self, N_items, N_recommended,behaviour="random", proba_p = 0.7 , name = 'envi_01'): #proba_p for ramdom choice (customer)
        self.items = Items(N_items)
        self.recommendation = Recommendation(self.items, N_recommended)
        self.customer = Customer( self.items, self.recommendation, behaviour, proba_p )
        self.name = name


    #self.step to simulate new step of the environnement
    def step(self, agentRecommendation): #here agentRecommendation is the items recommended by the agent
        self.recommendation.recommend(agentRecommendation) #We have set the new recommendations
        self.customer.choice()
        reward = self.computeReward()
        return reward



    def computeReward(self):#This function will be refined to get more realistic rewards
        reward = -self.items.items[self.customer.choice_id].cost
        reward += self.items.similarities[self.customer.previous_choice_id][self.customer.choice_id]
        return reward

    def display(self, print_item = False):
        print('---------ENVIRONNEMENT DISPLAY--------')
        self.items.display(print_item)
        self.recommendation.display()
        self.customer.display()

