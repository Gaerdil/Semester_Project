#Here, we simulate a customer (user) of the service.
#the customer wants to watch high quality content

import random
import numpy as np

class Customer():
    def __init__(self, items, recommendation, behaviour = 'random' , p = 0.7, specific_items = None): #p is the probability to trust
        #The customer has this way a direct access to
        self.items = items
        self.recommendation = recommendation
        if behaviour == 'specific':
            self.specific_items = specific_items
        #This would make us always choose the same beginning state.
        #self.previous_choice_id = -1
        #self.choice_id = -1 #it will be updated to the id of the next choice

        #Let's instead choose a random first state:
        self.previous_choice_id = random.randint(0, self.items.n_items -1)
        self.choice_id = random.randint(0, self.items.n_items -1)
        while   self.choice_id == self.previous_choice_id:
            self.choice_id = random.randint(0, self.items.n_items -1)

        self.trust_recommendation = False #If trust recommendations, this will be updated to True
        self.behaviour = behaviour
        self.p = p

        #List used for debugging
        self.choicesThisEpisode = np.zeros(self.items.n_items)

    def choice(self):
        if self.behaviour == 'random' :
            self.choiceRandom(self.p)

        elif self.behaviour == 'choiceFirst' :
            self.choiceFirst()

        elif self.behaviour == 'specific':
            self.choiceSpecific()

        elif self.behaviour == 'similar':
            self.choiceSimilar()

        elif self.behaviour == "randomMinSimilarQuality":
            self.choiceRandomMinSimilarQuality()

        else:
            print("Error: no choice method indicated")
        self.choicesThisEpisode[self.choice_id] += 1

    def endEpisode(self):
        self.previous_choice_id = random.randint(0, self.items.n_items -1)
        self.choice_id = random.randint(0, self.items.n_items -1)
        while   self.choice_id == self.previous_choice_id:
            self.choice_id = random.randint(0, self.items.n_items -1)
        self.trust_recommendation = False
        self.choicesThisEpisode = np.zeros(self.items.n_items)

    def display(self, print_item = False):
        print("----------- Customer -----------")
        print("Trust recommendation: "+str(self.trust_recommendation))
       # print(" Previous choice: "+str(self.previous_choice_id))
        print(" Choice: "+str(self.choice_id))
        if print_item:
            self.items.items[self.choice_id].display()

    #Here, we have a list of functions, to model customer behaviours
    # We will create other functions, to simulate a wider range of Customer behaviours (and more realistic)

    def choiceRandom(self, p): #a is the probability to pick something in the recommendations
        self.trust_recommendation = False
        self.previous_choice_id = self.choice_id
        b = random.random()
        if p >= b :
            self.trust_recommendation = True
            self.choice_id = random.choice(self.recommendation.recommended_items)
        else:
            while self.previous_choice_id == self.choice_id:
                self.choice_id = random.choice(self.items.ids)


    def choiceFirst(self): #this user will always choose the first item in the recommendations
        self.trust_recommendation = True
        self.previous_choice_id = self.choice_id
        self.choice_id = self.recommendation.recommended_items[0]

    def choiceSpecific(self): #this user will always choose the specific items
        self.trust_recommendation = False
        self.previous_choice_id = self.choice_id
        s = 0
        while s < len(self.specific_items) and not self.trust_recommendation:
            specific = self.specific_items[s]
            if specific in self.recommendation.recommended_items : #the current choiceId should not be there
                self.trust_recommendation = True
                self.choice_id = specific
            s += 1
        if not self.trust_recommendation:
            while self.choice_id == self.previous_choice_id:
                self.choice_id = random.choice(self.specific_items)

    def choiceSimilar(self): #this user has a minimum standard of "quality", being the similarities
        self.trust_recommendation = False
        self.previous_choice_id = self.choice_id
        for id in self.recommendation.recommended_items :
            if self.items.similarities[id][self.choice_id] >= self.p and not self.trust_recommendation:
                self.trust_recommendation = True
                self.choice_id = id
                break
        if not self.trust_recommendation: #Choose randomly among the 2 items with best similarity
            similarities = self.items.similarities[self.choice_id]
            self.choice_id = np.random.choice(similarities.argsort()[- 3 :])   #/!\ with big lists, problems...
            while self.choice_id == self.previous_choice_id :
                self.choice_id = np.random.choice(similarities.argsort()[- 3 :])

        # if not self.trust_recommendation: #Choose randomly among the 2 items with best similarity
        #     similarities = self.items.similarities[self.choice_id]
        #     self.choice_id = np.random.choice(similarities.argsort()[- 3 :])   #/!\ with big lists, problems...
        #     # while self.choice_id == self.previous_choice_id or self.items.similarities[self.previous_choice_id][self.choice_id] < self.p:
        #     #     self.choice_id = random.choice(self.items.ids)

    def choiceRandomMinSimilarQuality(self):
        b = random.random()
        if self.p >= b:
            self.choiceSimilar()
        else: #We pick randomly one of the three items with best similarity
            #Similarity on diagonal is set to - inf
            self.trust_recommendation = False
            self.previous_choice_id = self.choice_id
           # similarities = self.items.similarities[self.choice_id]
           # self.choice_id = np.random.choice(similarities.argsort()[- 3:])

            #Appeared to be too long/create bugs : if not good enough quality, a random item is picked.
            while self.previous_choice_id == self.choice_id:
                self.choice_id = random.choice(self.items.ids)











