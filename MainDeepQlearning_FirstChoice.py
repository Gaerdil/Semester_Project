from Items import *
from Recommendation import *
from Customer import *
from Environnement import *
from Agent import *
from Episode import *
from Series import *
from GridSearch import *
import matplotlib.pyplot as plt
import torch
from torch import nn

print(">>>>>>>>>>> TESTING THE AGENT : IN CASE THE CUSTOMER ALWAYS CHOOSES THE FIRST RECOMMENDATION <<<<<<<<<<<<<<<<<<")

# ------------ Defining several parameters - others will be chosen by grid search --------------
N_items = 10
N_recommended = 1
memory = 1
choiceMethod =  'DeepQlearning'
rewardType = 'Trust'
behaviour = 'choiceFirst'
rewardParameters = [1,-1]
steps = 50
epochs = 3
train_list = [True for u in range(3) ]+[ False, False ]


#------------- Defining the environnement  -----------
environnement = Environnement(N_items, N_recommended, behaviour,  rewardType , rewardParameters )

#>>> let's test the efficiency of our algorithm by testing with this simplified set:
for item in environnement.items.items :
    item.cost = 1
environnement.items.items[7].cost =0
#environnement.items.items[1].cost =0
#<<<

environnement.items.display(True)


#Create model
model = nn.Sequential(
    nn.Linear(memory+N_recommended, 20),
    nn.SELU(),
    nn.Linear(20, 5),
    nn.SELU(),
    nn.Linear(5, 1),
    nn.Sigmoid()

)
trainable_layers = [0,2,4]

deepQModel = {'model': model, 'trainable_layers': trainable_layers}

# >>> Grid search over the parameters to get the best parameters
gridSearch = GridSearch()
num_avg = 4
_ , params = gridSearch(num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=steps, more_params = None, deepQModel=deepQModel)

print("Testing the Grid Search parameters: ")

#------------ launching the episode series : Average the learning processes results   ---------------
#(less randomness in the plots), for statistical study, than the Series class
num_avg = 4
avgSeries = AverageSeries(num_avg, environnement, memory, choiceMethod, params, epochs, train_list, steps, deepQModel)
Rewards = avgSeries.avgRewards


plt.figure()
plt.plot(Rewards, 'r-')
plt.title("Average reward per serie")
plt.show()
#
#
