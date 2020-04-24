from Items import *
from Recommendation import *
from Customer import *
from Environnement import *
from Agent import *
from Episode import *
from Series import *
from GridSearch import *
import matplotlib.pyplot as plt

print(">>>>>>>>>>> TESTING THE AGENT : IN CASE THE CUSTOMER MAKES RANDOM CHOICES <<<<<<<<<<<<<<<<<<")
print("In this setting, the number of recommended items is 2. And only 2 items have a cost of 0. But the customer makes random choices. Will the agent adapt?")

# ------------ Defining several parameters - others will be chosen by grid search --------------
N_items = 10
N_recommended = 2
memory = 1
choiceMethod =  'QlearningActionsTuples'
rewardType = 'Trust'
behaviour = 'random'
rewardParameters = [1,1]
steps = 10
epochs = 5
display = False
displayItems  = False
train_list = [True for u in range(3) ]+[ False, False ]

#------------- Defining the environnement  -----------
environnement = Environnement(N_items, N_recommended, behaviour,  rewardType , rewardParameters )

#>>> let's test the efficiency of our algorithm by testing with this simplified set:
for item in environnement.items.items :
    item.cost = 1
environnement.items.items[2].cost =0
environnement.items.items[4].cost =0
#<<<

environnement.items.display(True)


# >>> Grid search over the parameters to get the best parameters
gridSearch = GridSearch()
num_avg = 3
_ , params = gridSearch(num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=steps)



#------------ launching the episode series : Average the learning processes results   ---------------
#(less randomness in the plots), for statistical study, than the Series class
num_avg = 5
display_avg = True
avgSeries = AverageSeries(num_avg, environnement, memory, choiceMethod, params, epochs, train_list, steps,display_avg , display, displayItems)
Rewards = avgSeries.avgRewards


plt.figure()
plt.plot(Rewards, 'r-')
plt.title("Average reward per serie")
plt.show()
#
#
