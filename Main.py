from Environnement import *
from GridSearch import *
import matplotlib.pyplot as plt

print(">>>>>>>>>>> TESTING THE AGENT : IN CASE THE CUSTOMER MAKES random choices (with min SIMILAR quality) <<<<<<<<<<<<<<<<<<")
print("Testing with more items: ")

# ------------ Defining several parameters - others will be chosen by grid search --------------
N_items = 100
N_recommended = 2
memory = 1
choiceMethod =  'QlearningActionsTuples'
rewardType = 'Trust'
behaviour = 'randomMinSimilarQuality'
rewardParameters = [1,1]
steps = 50
epochs = 5


train_list = [True for u in range(4) ]+[ False ]
p = 0.5

#------------- Defining the environnement  -----------
environnement = Environnement(N_items, N_recommended, behaviour,  rewardType , rewardParameters, proba_p=p )


# >>> Grid search over the parameters to get the best parameters
gridSearch = GridSearch()
num_avg = 3
_ , params = gridSearch(num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=steps)



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
