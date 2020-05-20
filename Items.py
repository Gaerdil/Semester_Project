#We define here a simplified environnement for our problem : Items part
#Just the list of items, with similarities and cost
import numpy as np
import random
from tqdm import *

class Items(): #part of the environnement directly dealing with items (database, costs, similariries)
    def __init__(self, N_items):

        #Creating the database of all of the items
        # (ex: one item is one song)
        self.n_items = N_items
        self.items = []
        self.ids = [] #we only have the ids of the items displayed here
        self.similarities = np.zeros((self.n_items,self.n_items))
        self.createItems()
        self.computeSimilarities()




    def createItems(self):
        for i in range(self.n_items):
            item = Item(i,random.randint(3,9))
            self.items.append(item)
            self.ids.append(i)


    def computeSimilarities(self):
        for i in tqdm(range(self.n_items)):
            for j in range(self.n_items):
                if i==j:
                    self.similarities[i][j]=1
                elif j>i:
                    self.similarities[i][j] = random.random()
                else:
                    self.similarities[i][j] = self.similarities[j][i]

    def display(self, print_items = False, print_similarities = False ):
        print("---------------- Items ----------------")
        print("Number of items: "+str(self.n_items))
        if print_items:
            print("*** Items list: ***")
            for item in self.items:
                item.display()
            print("***************")
        if print_similarities:
            print("*** Similarities: ***")
            print(self.similarities)
            print("***************")

class Item(): #a single item of the items database
    def __init__(self, id , size="7", binary = True, name = 'none'):

        self.id = id
        if name == 'none':
            self.name = RandomName(size)
        elif name != 'none':
            self.name = name

        if binary: #cached = 0, not cached = 1
            # Parameter can be changed (proportion of cached content)
            self.cost = int(np.random.choice([0,1], p = [0.05,0.95])) #Only a minority of items is cached
        elif binary == False: #real numbers for costs
            self.cost = 100*random.random()

    def display(self):
        print("Item "+str(self.id)+" -> name:" +self.name+", cost: "+str(self.cost))


def RandomName(size):
    alphabet = "AZERTYUIOPMLKJHGFDSQWXCVBNazertyuiopmlkjhgfdsqwxcvbn7894561230"
    name = ""
    for i in range(size):
        name = name + random.choice(alphabet)
    return name

