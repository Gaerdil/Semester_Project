# This file will help us compare results between algorithms
# Our algorithms are not yet too scalable to very large datasets because of the for loops
#We will use the similarWithSubset user, as it is the most 'realistic' one


from Environnement import *
from GridSearch import *
import matplotlib.pyplot as plt
from torch import nn



# -- environment parameters --
N_items = 10
N_recommended = 1
memory = 1
rewardType = 'Trust'
behaviour = 'similarWithSubset'
rewardParameters = [1,1]
steps = 10
epochs = 10
epochs_test = 20
num_avg = 3
train_list = [True for u in range(3) ]+[ False, False ]
min_similarities_sum = 0.7 #TODO : explore / change this user behaviour accordingly

def results_on_same_environnement():
    # ______ Environment creation _________
    environnement = Environnement(N_items, N_recommended, behaviour,  rewardType , rewardParameters , proba_p=min_similarities_sum )
    environnement.items.items[0].cost = 0 # To remove later
    # ______ RL agents ____________________

    # a) DeepQlearning
    # --params --
    choiceMethod =  'DeepQlearning'
    model = nn.Sequential(
        nn.Linear(memory+2*N_recommended, 10),
        nn.SELU(),
        nn.Linear(10, 1)
    )
    trainable_layers = [0,2]
    deepQModel = {'model': model, 'trainable_layers': trainable_layers}
    # -- Grid search the best parameters --
    gridSearch = GridSearch()
    _ , params = gridSearch(num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=steps, more_params = None, deepQModel=deepQModel)
    # -- Results with the best hyper parameters --
    avgSeries_DL = AverageSeries(num_avg, environnement, memory, choiceMethod, params, epochs_test, train_list, steps, deepQModel)
    Rewards_DeepQLearning = avgSeries_DL.avgRewards

    # b) Tabular Q Learning
    # --params --
    choiceMethod =  'QlearningActionsTuples'
    # -- Grid search the best parameters --
    gridSearch = GridSearch()
    _ , params = gridSearch(num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=steps)
    # -- Results with the best hyper parameters --
    avgSeries_Tabular = AverageSeries(num_avg, environnement, memory, choiceMethod, params, epochs_test, train_list, steps)
    Rewards_Tabular = avgSeries_Tabular.avgRewards

    # c) Linear Q Learning
    # --params --
    choiceMethod =  'LinearQlearning'
    # -- Grid search the best parameters --
    gridSearch = GridSearch()
    _ , params = gridSearch(num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=steps)
    # -- Results with the best hyper parameters --
    avgSeries_Linear = AverageSeries(num_avg, environnement, memory, choiceMethod, params, epochs_test, train_list, steps)
    Rewards_Linear = avgSeries_Linear.avgRewards

    return np.array(Rewards_Tabular), np.array(Rewards_DeepQLearning), np.array(Rewards_Linear)

# -- fetching the results --
Rewards_Tabular_avg, Rewards_DeepQLearning_avg, Rewards_Linear_avg = results_on_same_environnement()
for i in range(2) :
    Rewards_Tabular, Rewards_DeepQLearning, Rewards_Linear = results_on_same_environnement()
    Rewards_Tabular_avg = Rewards_Tabular_avg + Rewards_Tabular[:]
    Rewards_DeepQLearning_avg = Rewards_DeepQLearning_avg + Rewards_DeepQLearning[:]
    Rewards_Linear_avg = Rewards_Linear_avg + Rewards_Linear[:]

Rewards_Tabular_avg, Rewards_DeepQLearning_avg, Rewards_Linear_avg = Rewards_Tabular_avg /3, Rewards_DeepQLearning_avg/3, Rewards_Linear_avg /3
# -- plotting the results --

x_labels = [str(i)+"_"+str(train_list[i]) for i in range(len(train_list))]

plt.figure()
plt.xticks(range(len(x_labels)), x_labels, size='small')
plt.plot(Rewards_Tabular_avg, 'r-', label = "Tabular")
plt.plot(Rewards_DeepQLearning_avg, 'b-', label = "Deep Learning" )
plt.plot(Rewards_Linear_avg, 'g-', label = "Linear" )
plt.ylabel("Average reward per serie")
plt.xlabel("Serie id and type: true for training, false for testing  ")
plt.title("Average results of 3 environnements, of "+str(num_avg)+" parallel training/testing sessions per environnement, per algorithm type")
plt.legend()
plt.show()