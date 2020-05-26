
from Environnement import *
from GridSearch import *
import matplotlib.pyplot as plt
from torch import nn

print(">>>>>>>>>>> TESTING THE AGENT : Similar choice With Subset - Very Large dataset <<<<<<<<<<<<<<<<<<")

# ------------ Defining several parameters - others will be chosen by grid search --------------
N_items = 10000 #/!\ With "Similar" user, it would be too long to get more items...
N_recommended = 3
memory = 2
choiceMethod =  'DeepQlearningFaster'
rewardType = 'Trust'
behaviour = 'similarWithSubset'
rewardParameters = [1,1]
steps = 10
epochs = 3
train_list = [True for u in range(3) ]+[ False, False ]
more_params = {'debug' : False, 'subset_size':30}
min_similarities_sum = N_recommended/3 #TODO : explore / change this user behaviour accordingly
#------------- Defining the environnement  -----------
print("--> creating environnement")
environnement = Environnement(N_items, N_recommended, behaviour,  rewardType , rewardParameters , proba_p=min_similarities_sum )


#environnement.items.display(True)


#Create model
# model = nn.Sequential(
#     nn.Linear(memory+2*N_recommended, 20),
#     nn.SELU(),
#     nn.Linear(20, 5),
#     nn.SELU(),
#     nn.Linear(5, 1),
#     nn.Sigmoid()
#
# )
# trainable_layers = [0,2,4]


model = nn.Sequential(
    nn.Linear(memory+2*N_recommended, 10),
    nn.SELU(),
    nn.Linear(10, 1)
)

trainable_layers = [0,2]

deepQModel = {'model': model, 'trainable_layers': trainable_layers}

# >>> Grid search over the parameters to get the best parameters
print("--> Grid search")
gridSearch = GridSearch()
num_avg = 3
_ , params = gridSearch(num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=steps, more_params = more_params, deepQModel=deepQModel)

print("--> Testing the Grid Search parameters ")

#------------ launching the episode series : Average the learning processes results   ---------------
#(less randomness in the plots), for statistical study, than the Series class
num_avg = 3
epochs = 10
avgSeries = AverageSeries(num_avg, environnement, memory, choiceMethod, params, epochs, train_list, steps, deepQModel)
Rewards = avgSeries.avgRewards


plt.figure()
plt.plot(Rewards, 'r-')
plt.title("Average reward per serie")
plt.show()
#
#
