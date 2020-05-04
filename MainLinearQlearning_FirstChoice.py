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
N_items = 10
N_recommended = 1
memory = 1
choiceMethod =  'LinearQlearning'
rewardType = 'Trust'
behaviour = 'choiceFirst'
rewardParameters = [1,1]
steps = 10
epochs = 5
train_list = [True for u in range(3) ]+[ False, False ]

#------------- Defining the environnement  -----------
environnement = Environnement(N_items, N_recommended, behaviour,  rewardType , rewardParameters )

#>>> let's test the efficiency of our algorithm by testing with this simplified set:
for item in environnement.items.items :
    item.cost = 1
#environnement.items.items[2].cost =0
environnement.items.items[7].cost =0
#<<<

environnement.items.display(True)


# >>> Grid search over the parameters to get the best parameters
gridSearch = GridSearch()
num_avg = 3
_ , params = gridSearch(num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=steps)

#params = {"QLchoiceMethod": "eGreedy",
 #                                "epsilon": 0.3,
  #                              "learning_rate": 0.1,
   #                              "gamma": 0.5}


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
#As expected, the linear model is not efficient... We will need to add some non linearity in order to get better results
#Some solutions to get better results would be to have some "one hot encoding" for the inputs.
#However, "one hot encoding" in the case of more than 1000 states/ actions, is not very desirable...