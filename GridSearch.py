from Series import *
import time
from tqdm import tqdm

#Helper function for hyper parameter tuning

class GridSearch(): #helper function for hyper parameter tuning
    def __init__(self, display = True):
        self.display = display

    def __call__(self, num_avg, environnement, memory, choiceMethod, epochs, train_list, steps=5
                     ,display_avg=False, display=False, displayItems=False ):

        startTime = time.time()

        if choiceMethod == "Qlearning" or choiceMethod =="QlearningActionsTuples" or choiceMethod == "LinearQlearning":
            learning_rates = [ 0.001, 0.01, 0.1, 1]
            epsilons = [0.01, 0.05, 0.1, 0.2, 0.4]
            gammas = [0.1,0.3,0.5,0.7,0.9]

            best_reward = -np.inf
            for lr in tqdm(learning_rates):
                for eps in epsilons:
                    for g in gammas:
                        params = {"QLchoiceMethod": "eGreedy",
                                  "epsilon": eps,
                                  "learning_rate": lr,
                                  "gamma": g}

                        average_series = AverageSeriesNoTqdm(num_avg, environnement, memory, choiceMethod, params, epochs, train_list,
                                      steps=steps, display_avg=display_avg, display=display, displayItems=displayItems)

                        if average_series.avgLastReward > best_reward:
                            best_reward = average_series.avgLastReward
                            best_params = {"QLchoiceMethod": "eGreedy",
                                      "epsilon": eps,
                                      "learning_rate": lr,
                                      "gamma": g}
            if self.display:
                print("******** Grid Search results : *******")
                print("best_reward: "+str(best_reward))
                print("best parameters ")
                print(best_params)
                print("**************************************")
            return best_reward, best_params
        else:
            return None


        endTime = time.time()
        print(" \n \n Execution time of grid Search: " + str(endTime - startTime))






class AverageSeriesNoTqdm(): #Helpful to get a better unbiased statistical estimate of the efficiency of our model
    #We redo the learning process several times and average the results to reduce the amount of randomness in our results

    def __init__(self, num_avg, environnement, memory, choiceMethod, params, epochs, train_list, steps=5,display_avg = True, display=True, displayItems=False):

        self.choicesLastSerie_total = np.zeros(environnement.items.n_items)
        if display_avg:
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
            if choiceMethod == 'Qlearning' or choiceMethod == 'QlearningActionsTuples' or choiceMethod == 'LearningQlearning' :
                print(params)

        agent = Agent(environnement, memory, choiceMethod, params)
        series = Series(environnement, agent, epochs, train_list, steps, display, displayItems)
        self.avgRewards = np.array(series.allRewards[:])

        for a in range(num_avg):
            # We keep the exact same environnement, but reinitialize the Q-table (testing if we were just lucky in the learning process)
            agent = Agent(environnement, memory, choiceMethod, params)
            series = Series(environnement, agent, epochs, train_list, steps,  display, displayItems)
            self.avgRewards =  self.avgRewards  +  np.array(series.allRewards[:])
            self.choicesLastSerie_total= self.choicesLastSerie_total + series.choicesLastSerie
        self.avgRewards = self.avgRewards/num_avg
        self.avgLastReward = self.avgRewards[-1]



