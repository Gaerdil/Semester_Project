# A serie is composed of several episodes


from Episode import *
import time
from tqdm import tqdm
from Agent import *
import copy


class AverageSeries(): #Helpful to get a better unbiased statistical estimate of the efficiency of our model
    #We redo the learning process several times and average the results to reduce the amount of randomness in our results

    def __init__(self, num_avg, environnement, memory, choiceMethod, params, epochs, train_list, steps=5,deepQModel = None, display=True ):

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

        if choiceMethod == "DeepQlearning":
            agent.Qlearning.setModel(copy.deepcopy(deepQModel['model']), deepQModel['trainable_layers'])



        series = Series(environnement, agent, epochs, train_list, steps)



        self.avgRewards = np.array(series.allRewards[:])
        if choiceMethod != "random" and choiceMethod != "Qlearning" : #as Qlearning is an old version
            self.choicesLastSerieActionTuples_total = np.zeros(agent.Qlearning.numActions)

        for a in tqdm(range(num_avg)):
            # We keep the exact same environnement, but reinitialize the Q-table (testing if we were just lucky in the learning process)
            agent = Agent(environnement, memory, choiceMethod, params)

            #DeepQlearning setup --------------------------------------------------------------
            if choiceMethod == "DeepQlearning":
                agent.Qlearning.setModel(copy.deepcopy(deepQModel['model']), deepQModel['trainable_layers'])
            #----------------------------------------------------------------------------------
            #print("-------------------- BEFORE ----------------------------------")
            #agent.Qlearning.display()  # TODO : remove the print
            series = Series(environnement, agent, epochs, train_list, steps)
           # print("-------------------- AFTER -----------------------------------")
            #agent.Qlearning.display()  # TODO : remove the print
            self.avgRewards =  self.avgRewards  +  np.array(series.allRewards[:])
            self.choicesLastSerie_total= self.choicesLastSerie_total + series.choicesLastSerie

            if choiceMethod != "random" and choiceMethod != "Qlearning" :
                self.choicesLastSerieActionTuples_total = self.choicesLastSerieActionTuples_total + series.choiceslastSerieActionTuples


        self.avgRewards = self.avgRewards/num_avg
        self.avgLastReward = self.avgRewards[-1]


        if display:
            endTime = time.time()
            print(" \n \n Execution time: "+str(endTime - startTime))


            if choiceMethod == "QlearningActionsTuples":
                print("Qtable of the last series ------------------------------>")
                print(agent.Qlearning.Qtable)
                print("---------------------------------------------------->")
                print(" ")
                print("After the learning process : how often  is an item recommended? (total of all series) ")
                print(self.choicesLastSerie_total)
                print("After the learning process : how often  is an Action tuple recommended? (total of all series) ")
                print("Action list:")
                print(agent.Qlearning.actions)
                print("Action ids list:")
                print(agent.Qlearning.actions_ids)
                print("Number of time selected (per action id):")
                print(self.choicesLastSerieActionTuples_total)
                print("Most recommended action: " + str(agent.Qlearning.actions[np.argmax(np.array(self.choicesLastSerieActionTuples_total))]))

            elif choiceMethod == "LinearQlearning":
                print("Final weights: ")
                print(agent.Qlearning.weights)
                print("Action list:")
                print(agent.Qlearning.actions)
                print("Action ids list:")
                print(agent.Qlearning.actions_ids)
                print("Number of time selected (per action id):")
                print(self.choicesLastSerieActionTuples_total)
                print("Most recommended action: "+str(agent.Qlearning.actions[np.argmax(np.array(self.choicesLastSerieActionTuples_total))]))

            elif  choiceMethod == "PolynomialQlearning":
                print("Final weights: ")
                print(agent.Qlearning.weights)
                print("Action list:")
                print(agent.Qlearning.actions)
                print("Action ids list:")
                print(agent.Qlearning.actions_ids)
                print("Number of time selected (per action id):")
                print(self.choicesLastSerieActionTuples_total)
                print("Most recommended action: " + str(agent.Qlearning.actions[np.argmax(np.array(self.choicesLastSerieActionTuples_total))]))

            elif choiceMethod == "SimpleDeepQlearning":
                print(">>> Final weights: ")
                print("Hidden weights")
                print(agent.Qlearning.weights1)
                print("Hidden bias")
                print(agent.Qlearning.bias1)
                print("Output weights")
                print(agent.Qlearning.weights2)
                print("Hidden bias")
                print(agent.Qlearning.bias2)
                print("Action list:")
                print(agent.Qlearning.actions)
                print("Action ids list:")
                print(agent.Qlearning.actions_ids)
                print("Number of time selected (per action id):")
                print(self.choicesLastSerieActionTuples_total)
                print("Most recommended action: " + str(agent.Qlearning.actions[np.argmax(np.array(self.choicesLastSerieActionTuples_total))]))

            elif choiceMethod == "DeepQlearning":
                agent.Qlearning.display(True)
                print("Action list:")
                print(agent.Qlearning.actions)
                print("Action ids list:")
                print(agent.Qlearning.actions_ids)
                print("Number of time selected (per action id):")
                print(self.choicesLastSerieActionTuples_total)
                print("Most recommended action: " + str(agent.Qlearning.actions[np.argmax(np.array(self.choicesLastSerieActionTuples_total))]))

            print("------------------> Series ends <------------------")

class Series(): #several series, to show the whole learning/testing process
    def __init__(self, environnement, agent, epochs, train_list, steps=5,  display=False):

       # print("-----------BEGIN")
       # print(agent.Qlearning.weights)

        if display:
            print("------------------> Series begins <------------------")

        self.allRewards = []
        for train_ in train_list:
            serie = Serie(environnement, agent, epochs, steps, train_, display)
           # self.allRewards = self.allRewards + serie.serieRewards[:]
            self.allRewards.append(np.mean(serie.serieRewards[:]))
        self.choicesLastSerie = serie.choicesThisSerie[:] #Equal to the choices done at the last "not training" serie (end of the learning process)
        if agent.choiceMethod != "random" and agent.choiceMethod != "Qlearning":
            self.choiceslastSerieActionTuples = serie.choicesThisSerieActionTuples

        if display:
            print("------------------> Series ends <------------------")
        #print("-----------END")
        #print(agent.Qlearning.weights)


class Serie(): #a serie is a serie of episodes with all the same train_ type (true or false).
    # train_ indicates if the agent is going to updates its Qtable during the episode (training)
    def __init__(self, environnement, agent, epochs, steps = 5, train_ = False, display = False ):
        self.serieRewards = []
        self.choicesThisSerie = np.zeros(environnement.items.n_items)

        if display :
            self.display(train_)

        if agent.choiceMethod != "random" and agent.choiceMethod != "Qlearning":
            self.choicesThisSerieActionTuples = np.zeros(agent.Qlearning.numActions)

        for epoch in range(epochs ):
            episode = Episode(environnement, agent, steps, train_)
            self.serieRewards.append(np.mean(episode.episodeReward)) #Taking the mean should help get more consistent results
            self.choicesThisSerie = self.choicesThisSerie + episode.choicesThisEpisode
            if agent.choiceMethod != "random" and agent.choiceMethod != "Qlearning":
                self.choicesThisSerieActionTuples = self.choicesThisSerieActionTuples + episode.choicesThisEpisodeActionTuples

    def display(self, train_):
        print("------------------> Serie begins")
        print("Training session: "+str(train_))



        #TODO : paragraph that could be a good starting point for cloning model in AverageSeries, in case copy.deepCopy does not work well with python version
        #self.model =deepQModel['model']
            #self.trainable_layers = deepQModel['trainable_layers']
            #Lets keep the model's parameter, in order to get the same initialization at each average:
            #self.initial_deepQ_model_values = {}
            #for i in self.trainable_layers:
                #self.initial_deepQ_model_values['w'+str(i)]  =  self.model[i].weight.data.clone()
                #self.initial_deepQ_model_values['b' + str(i)] = self.model[i].bias.data.clone()
            #setting the model to the agent