from Series import *
import time
from tqdm import tqdm

#Helper function for hyper parameter tuning

class GridSearch(): #helper function for hyper parameter tuning
    def __init__(self, display = True):
        self.display = display


    def __call__(self, num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=5, more_params = None, deepQModel =None):

        startTime = time.time()

        if choiceMethod != "random" :
            learning_rates = [ 0.001, 0.01, 0.1, 1]
            #epsilons = [0.01, 0.05, 0.1, 0.2, 0.4]
            epsilons = [ 0.1, 0.2, 0.4]
            gammas = [0.1,0.3,0.5,0.7,0.9]


  # ----- AVOIDING NAN VALUES or Vanishing gradients ------------------------------------

            if choiceMethod == "SimpleDeepQlearning":
                learning_rates = [ 0.001, 0.01, 0.1]
                epsilons = [0.1, 0.3]
                gammas = [0.1,  0.5,  0.9]

            if choiceMethod == "DeepQlearning" or choiceMethod == "DeepQlearningFaster":
                learning_rates = [1e-3, 1e-2, 1e-1]
                epsilons = [0.1,0.3]
                gammas = [0.1,  0.5,  0.9]
  # --------------------------------------------------------------


            params = {"QLchoiceMethod": "eGreedy",
                      "epsilon": None,
                      "learning_rate": None,
                      "gamma": None}

            best_params = {"QLchoiceMethod": "eGreedy",
                           "epsilon": None,
                           "learning_rate": None,
                           "gamma": None}

            if choiceMethod == "PolynomialQlearning":
                params['degree']= more_params['degree']
                best_params['degree'] = more_params['degree']

            if choiceMethod == "SimpleDeepQlearning" :
                params['hidden_size']= more_params['hidden_size']
                best_params['hidden_size'] = more_params['hidden_size']

            if choiceMethod == "DeepQlearningFaster":
                params['subset_size'] = more_params['subset_size']
                best_params['subset_size'] = more_params['subset_size']
                params['debug'] = more_params['debug']
                best_params['debug'] = more_params['debug']


            best_reward = -np.inf
            for lr in tqdm(learning_rates):
                #for g in tqdm(gammas):
                for g in gammas:
                    #for eps in tqdm(epsilons):
                    for eps in epsilons:

                        #params = {"QLchoiceMethod": "eGreedy",
                         #         "epsilon": eps,
                          #        "learning_rate": lr,
                          #        "gamma": g}

                        params["epsilon"], params['learning_rate'], params['gamma'] = eps,lr,g

                        average_series = AverageSeriesNoTqdm(num_avg, environnement, memory, choiceMethod, params, epochs, train_list,
                                      steps=steps, deepQModel = deepQModel)

                        if average_series.avgLastReward > best_reward:
                            best_reward = average_series.avgLastReward
                            best_params["epsilon"], best_params['learning_rate'], best_params['gamma'] = eps, lr, g


            endTime = time.time()
            if self.display:
                print("******** Grid Search results : *******")
                print("best_reward: "+str(best_reward))
                print("best parameters ")
                print(best_params)
                print("**************************************")
                print(" \n \n Execution time of grid Search: " + str(endTime - startTime))

            return best_reward, best_params
        else:
            return None









class AverageSeriesNoTqdm(): #Helpful to get a better unbiased statistical estimate of the efficiency of our model
    #We redo the learning process several times and average the results to reduce the amount of randomness in our results

    def __init__(self, num_avg, environnement, memory, choiceMethod, params, epochs, train_list, steps=5,deepQModel = None, display = False):

        self.choicesLastSerie_total = np.zeros(environnement.items.n_items)
        if display:
            startTime = time.time()
            print("------------------> Average of series begins:  <------------------")
            print(str(num_avg)+" independent training/testing processes")
            print("environnement name: "+environnement.name)
            print("Memory size: " + str(memory))
            print("Number of items to recommend: "+ str(environnement.recommendation.n_recommended))
            print("--- We will test the following hyperparameters ---")
            print("choice method: " + choiceMethod)
            print("epochs: "+ str(epochs))
            print("Reward hyper parameters: "+ str(environnement.rewardParameters))
            if choiceMethod != "random" :
                print(params)

        agent = Agent(environnement, memory, choiceMethod, params)
        if choiceMethod == "DeepQlearning" or choiceMethod == "DeepQlearningFaster":
            agent.Qlearning.setModel(copy.deepcopy(deepQModel['model']), deepQModel['trainable_layers'])

        series = Series(environnement, agent, epochs, train_list, steps)
        self.avgRewards = np.array(series.allRewards[:])

        for a in range(num_avg):
            # We keep the exact same environnement, but reinitialize the Q-table (testing if we were just lucky in the learning process)
            agent = Agent(environnement, memory, choiceMethod, params)
            # DeepQlearning setup --------------------------------------------------------------
            if choiceMethod == "DeepQlearning" or choiceMethod == "DeepQlearningFaster":
                agent.Qlearning.setModel(copy.deepcopy(deepQModel['model']), deepQModel['trainable_layers'])
            # ----------------------------------------------------------------------------------
            series = Series(environnement, agent, epochs, train_list, steps)
            self.avgRewards =  self.avgRewards  +  np.array(series.allRewards[:])
            self.choicesLastSerie_total= self.choicesLastSerie_total + series.choicesLastSerie
        self.avgRewards = self.avgRewards/num_avg
        self.avgLastReward = self.avgRewards[-1]



