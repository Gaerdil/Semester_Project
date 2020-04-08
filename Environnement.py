import random
from Items import *
from Recommendation import *
from Customer import *

class Environnement():
    def __init__(self, N_items, N_recommended,behaviour="random", proba_p = 0.7 , rewardType = 'Similarity', rewardParameters = [1,1] ,name = 'envi_01'): #proba_p for ramdom choice (customer)
        self.items = Items(N_items)
        self.recommendation = Recommendation(self.items, N_recommended)
        self.customer = Customer( self.items, self.recommendation, behaviour, proba_p )
        self.name = name
        self.rewardType = rewardType
        self.rewardParameters = rewardParameters

    def endEpisode(self):
        self.customer.endEpisode()
        self.recommendation.endEpisode()

    #self.step to simulate new step of the environnement
    def step(self, agentRecommendation): #here agentRecommendation is the items recommended by the agent
        self.recommendation.recommend(agentRecommendation) #We have set the new recommendations
        self.customer.choice()
        reward = self.computeReward()
        return reward



    def computeReward(self):#This function will be refined to get more realistic rewards
        if self.rewardType == 'Similarity':
            reward = -self.items.items[self.customer.choice_id].cost * self.rewardParameters[0]
            reward += self.items.similarities[self.customer.previous_choice_id][self.customer.choice_id] * self.rewardParameters[1]
        elif self.rewardType == 'Trust':
            reward = -(self.items.items[self.customer.choice_id].cost )* self.rewardParameters[0]
            reward += self.customer.trust_recommendation * self.rewardParameters[1]
        else:
            print("Error : wrong reward type")
            return None

        return reward

    def display(self, print_item = False):
        print('---------ENVIRONNEMENT DISPLAY--------')
        self.items.display(print_item)
        self.recommendation.display()
        self.customer.display()

