#Here, we simulate a customer (user) of the service.
#the customer wants to watch high quality content

import random
import numpy as np

class Customer():
    def __init__(self, items, recommendation, behaviour = 'random' , p_or_minSum = 0.7, specific_items = None): #p is the probability to trust
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

        if self.behaviour == 'similarWithSubset':
            self.min_similarity_sum = p_or_minSum #in this case, the p_or_minSum variable can be any real number

        else:
            self.p = p_or_minSum
            #Here p is a probability that has to be between 0 and 1 ... the lines below are ensuring that
            if self.p <0 :
                self.p =0
            elif self.p >1 :
                self.p = 1

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

        elif self.behaviour == "similarWithSubset":
            self.choiceSimilarWithSubset()

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


    def choiceSimilarWithSubset(self): #This user behaviour will be usable with the deep learning "faster" method
        #Contrary to "Similar" user, it is scalable to huge catalogs of items
        recommendation_similarities = [self.items.similarities[id][self.choice_id] for id in self.recommendation.recommended_items ]
        recommendation_similarities_sum = np.sum(np.array(recommendation_similarities))
        #print(self.min_similarity_sum, recommendation_similarities_sum)
        #TODO : set this to mean instead later maybe?
        if self.min_similarity_sum <= recommendation_similarities_sum :
            #The recommendation is globally good enough:
            self.trust_recommendation = True #We will pick an item into the recommendations
            self.choice_id = int(np.random.choice(self.recommendation.recommended_items, 1, p =softmaxOfSims(recommendation_similarities)))
            #We will pick this item with the similarities as the probabilities (with softmax transformation)

        else:
            self.trust_recommendation = False
            self.previous_choice_id = self.choice_id

    #To adapt to huge dataset, we will take the best option out of a subset of items
    #Indeed, this is more realistic: a user would not see all of the descriptions of the items  in youtube before making a choice.
    #The user would rather have a look at a smaller subset and make a choice within this smaller subset
            random_subset = np.random.choice(self.items.ids, min(self.items.n_items, 10), replace=False) #Random item ids
            #TODO : change the random subset size later
            random_subset_similarities = np.array([self.items.similarities[id][self.choice_id] for id in random_subset]) #The similarities of the random items ids
            while self.previous_choice_id == self.choice_id:  # To ensure we wont take two time the same item
                random_item_sim = np.random.choice(random_subset_similarities.argsort()[- 3:])
                self.choice_id =  int(random_subset[random_item_sim])
        #We randomly took one of the 3 items with the best similarity, in the random subset of items


#helper function :

def softmaxOfSims(similarities): #to change similarities into probabilities
    np_sims = np.array(similarities)
    exponential_similarities = np.exp(np_sims)
    return list(exponential_similarities/np.sum(exponential_similarities))







