#Here, we simulate a customer (user) of the service.
#the customer wants to watch high quality content

import random

class Customer():
    def __init__(self, items, recommendation, behaviour = 'random' , p = 0.7): #p is the probability to trust
        #The customer has this way a direct access to
        self.items = items
        self.recommendation = recommendation
        self.previous_choice_id = -1
        self.choice_id = -1 #it will be updated to the id of the next choice
        self.trust_recommendation = False #If trust recommendations, this will be updated to True
        self.behaviour = behaviour
        self.p = p

    def choice(self):
        if self.behaviour == 'random' :
            self.choiceRandom(self.p)
        else:
            print("Error: no choice method indicated")

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
            self.choice_id = random.choice(self.items.ids)


