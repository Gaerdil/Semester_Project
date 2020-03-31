import random
import numpy as np

class Episode():
    def __init__(self, environnement, agent, steps = 5, train_ = False, display = True , displayItems  = False):
        self.environnement = environnement
        self.agent = agent
        self.train_ = train_

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
            print(">>>>>>>>>>>>>> end  <<<<<<<<<<<<<<")

        #________________ Ending the episode : keep the total reward andreinitialize the agent
        self.episodeReward = self.agent.totalReward #We will
        self.agent.endEpisode()


    def step(self):
        self.agent.recommend()
        reward = self.environnement.step(self.agent.recommendation)
        self.agent.updateStateAndReward(reward)
        if self.train_ :
            self.agent.train()







