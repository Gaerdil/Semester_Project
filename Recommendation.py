class Recommendation(): #part of the environnement directly dealing with recommended items
    #no agent/AIfor the moment. The choice of the ids will be determined by the agent.
    def __init__(self,items, N_recommended): #items is an Items object, N_recommended the number of recommendations

        #About the recommended items
        #(the list will be updated)
        self.n_recommended =   N_recommended
        self.recommended_items = [] #list of ids of recommended items
        self.all_items = items #list of ALL items

    def recommend(self, ids): #assume we have a list of ids, we just update the list
        if len(ids) != self.n_recommended:
            print("ERROR : you must recommend exactly "+str(self.n_recommended)+ " items.")
        else:
            self.recommended_items = ids

    def display(self, print_items = False):
        print("---------------- Recommendations ----------------")
        print("Number of recommended items: "+str(self.n_recommended))
        if print_items:
            print("*** Items list: ***")
            for item_id in self.recommended_items:
                self.all_items.items[item_id].display()
            print("***************")