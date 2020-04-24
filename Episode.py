import random
import numpy as np

class Episode():
    def __init__(self, environnement, agent, steps = 5, train_ = False, display = True , displayItems  = False):
        self.environnement = environnement
        self.agent = agent
        self.train_ = train_
        self.agent.init_State()

        if display:
            print(">>>>>>>>>>>>>>>>> Episode  <<<<<<<<<<<<<<<<<")
            print('Length of the episode : '+str(steps))
            print('Agent name : '+self.agent.name+' , Method: '+self.agent.choiceMethod)
            print('Environnement name: '+self.environnement.name)
            self.environnement.display(displayItems)
            self.agent.display()

        for step in range(steps):
            self.step()  # new step of the environnement and agent
            if display:
                print(">>>>>>>>>> STEP "+str(step)+":")
                self.agent.display()
                self.environnement.customer.display()
        if display:
            print("List of the recommender choices:")
            print(self.environnement.recommendation.choicesThisEpisode)
            print("List of the choices of the customer :")
            print(self.environnement.customer.choicesThisEpisode)
            print(">>>>>>>>>>>>>> end  <<<<<<<<<<<<<<")

        #________________ Ending the episode : keep the total reward andreinitialize the agent
      #  print("WWWWWW" +str(self.environnement.recommendation.choicesThisEpisode))
       # print(self.environnement.customer.choicesThisEpisode)
        self.choicesThisEpisode = self.environnement.recommendation.choicesThisEpisode[:]
        if self.agent.choiceMethod == "QlearningActionsTuples":
            self.choicesThisEpisodeActionTuples = self.agent.choicesThisEpisode[:]
        self.episodeReward = self.agent.totalReward #We will
        self.agent.endEpisode()
        self.environnement.endEpisode()



    def step(self):
        if self.agent.choiceMethod == "QlearningActionsTuples":
            self.agent.recommend(self.train_)
        elif self.agent.choiceMethod != "QlearningActionsTuples":  #/!\ temporary : all agents should in the end also only choose the actions with maw value in non train mode
            self.agent.recommend()

        reward = self.environnement.step(self.agent.recommendation)
        self.agent.updateStateAndReward(reward)
        if self.train_ :
            self.agent.train()








