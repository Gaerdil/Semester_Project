from Items import *
from Recommendation import *
from Customer import *
from Environnement import *
from Agent import *
from Episode import *
from Series import *
from GridSearch import *
import matplotlib.pyplot as plt
import  torch
from torch import nn

print(">>>>>>>>>>> TEST TO REMOVE  <<<<<<<<<<<<<<<<<<")
print("In this setting, the number of recommended items is 2. And only 2 items have a cost of 0. Will the agent adapt?")

# ------------ Defining several parameters - others will be chosen by grid search --------------
N_items = 5
N_recommended = 1
memory = 1
choiceMethod =  'DeepQlearning'
rewardType = 'Trust'
behaviour = 'choiceFirst'
rewardParameters = [1,1]
steps = 5
epochs = 3

train_list = [False] #[True for u in range(3) ]+[ False, False ]

#------------- Defining the environnement  -----------
environnement = Environnement(N_items, N_recommended, behaviour,  rewardType , rewardParameters )

#>>> let's test the efficiency of our algorithm by testing with this simplified set:
for item in environnement.items.items :
    item.cost = 1
environnement.items.items[3].cost =0
#environnement.items.items[4].cost =0
#<<<

#environnement.items.display(True)

# >>> Grid search over the parameters to get the best parameters
#gridSearch = GridSearch()
#num_avg = 3
#_ , params = gridSearch(num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=steps)

params = {"QLchoiceMethod": "eGreedy",
                                  "epsilon": 0.1,
                                  "learning_rate": 0.1,
                                  "gamma": 0.5,
          }

agent = Agent(environnement, memory ,  choiceMethod ,  params)
model = nn.Sequential(
    nn.Linear(agent.Qlearning.n_inputs, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
trainable_layers = [0,2,4]
agent.Qlearning.setModel(model, trainable_layers)
print(">>>>>>>>>>> old weights <<<<<<<<<<<<<<")
agent.Qlearning.display()
print(" --> Old values")
fig, ax = plt.subplots(2,environnement.items.n_items)
for a in environnement.items.ids:
    print("actions/values for item "+str(a)+": "+str([agent.Qlearning.getValue([a], [b]) for b in environnement.items.ids])+" best_action: "+str(np.argmax(np.array([agent.Qlearning.getValue([a], [b]) for b in environnement.items.ids]))))
    ax[0][a].plot([agent.Qlearning.getValue([a], [b]) for b in environnement.items.ids])
    ax[0][a].set_title("b_"+str(a))

#Episode( environnement, agent, 5, False, False)
#Episode( environnement, agent, 10, True, False)
#Episode( environnement, agent, 5, False, False)
Serie(environnement, agent, epochs = 5, steps = 10, train_ = True)

print(">>>>>>>>>>> new weights <<<<<<<<<<<<<<")
agent.Qlearning.display()
print(" --> New values")
for a in environnement.items.ids:
    print("actions/values for item "+str(a)+": "+str([agent.Qlearning.getValue([a], [b]) for b in environnement.items.ids])+" best_action: "+str(np.argmax(np.array([agent.Qlearning.getValue([a], [b]) for b in environnement.items.ids]))))
    ax[1][a].plot([agent.Qlearning.getValue([a], [b]) for b in environnement.items.ids])
    ax[1][a].set_title("a_" + str(a))
plt.show()

#avgSeries = AverageSeries(3, environnement, memory, choiceMethod, params, epochs, train_list, steps, False, False, False)
