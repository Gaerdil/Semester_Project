from Items import *
from Recommendation import *
from Customer import *
from Environnement import *
from Agent import *
from Episode import *
from Series import *
import matplotlib.pyplot as plt



# ------------ Defining parameters --------------

N_items = 4
N_recommended = 2
memory = 2
choiceMethod = 'Qlearning'
behaviour = "random"
steps = 50
epochs = 10
display = False
displayItems  = False
train_list = [False, True, True, True, False]
params = {"QLchoiceMethod" : "eGreedy",
"epsilon" : 0.15,
"learning_rate" : 0.4,
          "gamma": 0.8 }

#------------- Defining the environnement and the agent -----------
environnement = Environnement(N_items, N_recommended, behaviour, proba_p = 0.7 )
agent = Agent(environnement, memory ,  choiceMethod ,  params )

#------------ launching the episode series (and keeping the results) ---------------
series = Series(environnement, agent, epochs, train_list, steps, display, displayItems)
Rewards = series.allRewards

plt.figure()
plt.plot(Rewards)
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
