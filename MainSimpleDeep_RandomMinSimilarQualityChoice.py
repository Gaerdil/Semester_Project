
from Environnement import *
from GridSearch import *
import matplotlib.pyplot as plt

print(">>>>>>>>>>> TESTING THE AGENT : IN CASE THE CUSTOMER MAKES SIMILAR CHOICES <<<<<<<<<<<<<<<<<<")
print("In this setting, the number of recommended items is 2. And only 2 items have a cost of 0. But the customer makes similar choices. Will the agent adapt?")

# ------------ Defining several parameters - others will be chosen by grid search --------------
N_items = 50
N_recommended = 1
memory = 2
choiceMethod =  'SimpleDeepQlearning'
rewardType = 'Trust'
behaviour = 'randomMinSimilarQuality'
rewardParameters = [1,1]
steps = 20
epochs = 3
train_list = [True for u in range(3) ]+[ False, False ]
p = 0.5
more_parameters = {'hidden_size':10}

#------------- Defining the environnement  -----------
environnement = Environnement(N_items, N_recommended, behaviour,  rewardType , rewardParameters, proba_p=p )
environnement.items.display()


# >>> Grid search over the parameters to get the best parameters
gridSearch = GridSearch()
num_avg = 3
_ , params = gridSearch(num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=steps, more_params = more_parameters)
#params = {'gamma': 0.1, 'hidden_size': 10, 'epsilon': 0.2, 'learning_rate': 0.01, 'QLchoiceMethod': 'eGreedy'}


#------------ launching the episode series : Average the learning processes results   ---------------
#(less randomness in the plots), for statistical study, than the Series class
num_avg = 3
avgSeries = AverageSeries(num_avg, environnement, memory, choiceMethod, params, epochs, train_list, steps)
Rewards = avgSeries.avgRewards


plt.figure()
plt.plot(Rewards, 'r-')
plt.title("Average reward per serie")
plt.show()
#
#
#Works for small settings...