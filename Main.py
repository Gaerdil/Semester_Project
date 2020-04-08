from Items import *
from Recommendation import *
from Customer import *
from Environnement import *
from Agent import *
from Episode import *
from Series import *
import matplotlib.pyplot as plt



# ------------ Defining parameters --------------
N_items = 5
N_recommended = 3
memory = 2
choiceMethod = 'Qlearning'
proba_p = 0.7
rewardType = 'Trust'

behaviour = 'choiceFirst'
rewardParameters = [1,1]     #With this set the agent manages to recommend the lowest cost item first

#behaviour = 'random'  #Even with this behaviour, we can see that the recommender will more often recommend the costless items

steps = 10
epochs = 50
display = False
displayItems  = False
train_list = [True for u in range(3) ]+[ False, False ]
params = {"QLchoiceMethod" : "eGreedy",
"epsilon" : 0.2,
"learning_rate" : 0.1,
          "gamma": 0.5 }

#------------- Defining the environnement and the agent -----------
environnement = Environnement(N_items, N_recommended, behaviour, proba_p , rewardType , rewardParameters )

#>>> let's test the efficiency of our algorithm by testing with this simplified set:
for item in environnement.items.items :
    item.cost = 1
environnement.items.items[2].cost =0
#<<<

environnement.items.display(True)



#agent = Agent(environnement, memory ,  choiceMethod ,  params )

#------------ launching the episode series (and keeping the results) ---------------
#The Series algo is enough to train the agent.
#series = Series(environnement, agent, epochs, train_list, steps, display, displayItems)
#Rewards = series.allRewards

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

# items_test = Items(6)
# items_test.display(True,True)
#
# recommend_test = Recommendation(items_test,2)
# recommend_test.recommend([1,4])
# recommend_test.display(True)
#
# customer_test = Customer(items_test, recommend_test)
# customer_test.display()
#
# customer_test.choiceRandom(0.6)
# customer_test.display(True)

