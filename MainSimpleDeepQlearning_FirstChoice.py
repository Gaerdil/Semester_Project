from Items import *
from Recommendation import *
from Customer import *
from Environnement import *
from Agent import *
from Episode import *
from Series import *
from GridSearch import *
import matplotlib.pyplot as plt

print(">>>>>>>>>>> TESTING THE AGENT : IN CASE THE CUSTOMER ALWAYS CHOOSES THE FIRST RECOMMENDATION <<<<<<<<<<<<<<<<<<")
print("In this setting, the number of recommended items is 2. And only 2 items have a cost of 0. Will the agent adapt?")

# ------------ Defining several parameters - others will be chosen by grid search --------------
N_items = 5
N_recommended = 1
memory = 1
choiceMethod =  'SimpleDeepQlearning'
rewardType = 'Trust'
behaviour = 'choiceFirst'
rewardParameters = [1,0]
steps = 30
epochs = 5
train_list = [True for u in range(3) ]+[ False, False ]
more_parameters = {'hidden_size':50}

#------------- Defining the environnement  -----------
environnement = Environnement(N_items, N_recommended, behaviour,  rewardType , rewardParameters )

#>>> let's test the efficiency of our algorithm by testing with this simplified set:
for item in environnement.items.items :
    item.cost = 1
environnement.items.items[3].cost =0
environnement.items.items[1].cost =0
#<<<

environnement.items.display(True)


# >>> Grid search over the parameters to get the best parameters
gridSearch = GridSearch()
num_avg = 5
_ , params = gridSearch(num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=steps, more_params = more_parameters)


#------------ launching the episode series : Average the learning processes results   ---------------
#(less randomness in the plots), for statistical study, than the Series class
num_avg = 5
avgSeries = AverageSeries(num_avg, environnement, memory, choiceMethod, params, epochs, train_list, steps)
Rewards = avgSeries.avgRewards


plt.figure()
plt.plot(Rewards, 'r-')
plt.title("Average reward per serie")
plt.show()
#
#
#Sometimes works a little bit. However: the model might be too simple, and there are quite some hyperparameters to change...