import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from hpbandster.optimizers import BOHB, randomsearch
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker
import gym
import random
import numpy as np
from pandas import DataFrame
from keras import models, layers
from keras.optimizers import Adam
import collections 
import matplotlib.pyplot as plt
import argparse
import pickle
import os
import logging
#logging.basicConfig(level=logging.DEBUG)

class MCDQNWorker(Worker):
    def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.env = gym.make('MountainCar-v0')
            self.replayBuffer = collections.deque(maxlen=20000)
                        
    def createNetwork(self, config):
            '''
            Sequential network with fully connected layers.
            '''
            model = models.Sequential()
            model.add(layers.Dense(config['L1'], activation='relu', input_shape=self.env.observation_space.shape))
            model.add(layers.Dense(config['L2'], activation='relu'))
            model.add(layers.Dense(self.env.action_space.n,activation='linear'))
            if config['optimizer'] == 'Adam':
                    optimizer = keras.optimizers.Adam(lr=config['lr'])
            else:
                    optimizer = keras.optimizers.SGD(lr=config['lr'], momentum=config['sgd_momentum'])
            model.compile(loss='mse', optimizer=optimizer,metrics=['accuracy'])
            return model

    def train_DQN(self, samplesize):
            '''
            Training the main_net using q learning, experience play, and infrequent weight update of clone_net or target net
            ## 1 Takes random 'n' samples from replay buffer
            ## 2 Predicts the best action based on the training network
            ## 3 Using bellman equation, for the states with game not done
            ## 4 Training the network
            '''
            states = DataFrame(random.sample(self.replayBuffer,samplesize))
            ## Getting the states, action, reward
            state, next_state  = np.array(list(states[0].to_numpy())),  np.array(list(states[3].to_numpy()))
            action, reward = states[1].to_numpy(), states[2].to_numpy()
            ## Calculating the Q values using the bellman equation
            qTable = self.main_net.predict(state)
            qTable[range(samplesize), action] = reward + self.discount * np.max(self.clone_net.predict(next_state), axis=1)
            ## Training the main network
            self.main_net.fit(state, qTable, epochs=1, verbose=False)

    def game(self, currentState, decay_rate, n=10):
            '''
            Net plays an individual games with move cap at 200 moves and a decay rate of 0.01 by default, returns the score of each game and the step at which the game was over
            ## 1 Move selection based on epsilon greedy strategy
            ## 2 Reward adjustment when the car reaches the top
            ## 3 Network is trained after the replay buffer size is more than n games
            ## 4 Infrequent weight updates inorder to deal with divergence        
            '''
            score = 0
            done = False
            while not done:
                ## Epsilon greedy strategy
                self.epsilon = max(self.epsilon_final, self.epsilon)
                if np.random.rand(1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action=np.argmax(self.main_net.predict(currentState.reshape(1,2)))
                ## Play the move returned by the epsilon greedy strategy
                new_state, reward, done, info = self.env.step(action)
                ## Reward for wins
                if new_state[0] >= 0.5:
                    reward += 10
                ## Saving the state transitions, action, reward and game status
                self.replayBuffer.append([currentState, action, reward, new_state, done])
                ## Training only after n games; Mini-batch
                if len(self.replayBuffer) > n: 
                    self.train_DQN(n)
                ## Total score
                score += reward
                currentState = new_state
            ## Infrequent weight updates   
            self.clone_net.set_weights(self.main_net.get_weights())
            ## Epsilon decay
            self.epsilon -= decay_rate
            return score, currentState[0]

    def compute(self, config, budget, working_directory, *args, **kwargs):
            """
            Uses two fully connected networks.
            main net is trained for allocated budget makes at most 200 moves.
            Returns the average reward.
            """
            self.main_net = self.createNetwork(config)
            ## Clone the network for infrequent weight updates
            self.clone_net = self.createNetwork(config)  ## Target network
            self.clone_net.set_weights(self.main_net.get_weights()) ## Setting the weights
            
            self.discount = config['discount'] ## Lower value = less important future values. Range [0,1]
            self.learningRate = config['lr']
            self.epsilon = config['tradeoff'] ## High epsilon high exploration, low epsilon more smart moves
            self.epsilon_final = 0.01 #config['ep_final'] ## Final epsilon value after decay 
            scores=[]
            final_position = []
            for e in range(int(budget)):
                start_state=self.env.reset()
                s, fp = self.game(start_state, config['decay']) ## Playing the game
                scores.append(s)
                final_position.append(fp)
            scores = np.mean(scores)
            final_position = np.mean(final_position) 

            ## Evaluation Run.
            testScores = []
            testFP = []
            for i in range(10):
                done = False
                currentState = self.env.reset()
                score=0
                step = 0
                while not done:
                    #env.render()  ## Comment it to hide the visual  window
                    action = np.argmax(self.main_net.predict(currentState.reshape(1,2)))
                    new_state, reward, done, info = self.env.step(action)
                    currentState=new_state
                    score+=reward
                    step+=1
                #print("Episode finished after {} timesteps reward is {}".format(step,score))
                testScores.append(score)
                testFP.append(currentState[0])

            ## Evaluation Results
            test_score = np.mean(testScores)
            #import IPython; IPython.embed()
            return ({
                    'loss': 0 - test_score, # remember: HpBandSter always minimizes!
                    'info': {'budget': int(budget),
                             'config': config,
                             'train_score': scores,
                             'train_FP': final_position,
                             'test_score': test_score,
                             'test_FP': testFP
                            }

            })


    @staticmethod
    def get_configspace():
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            cs = CS.ConfigurationSpace()

            lr = CSH.UniformFloatHyperparameter('lr', lower=1e-3, upper=1e-1, default_value='1e-2', log=True)

            # For demonstration purposes, we add different optimizers as categorical hyperparameters.
            # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
            # SGD has a different parameter 'momentum'.
            optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

            sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

            cs.add_hyperparameters([lr, optimizer, sgd_momentum])

            L1 = CSH.UniformIntegerHyperparameter('L1', lower=40, upper=60, default_value=54, log=True)
            L2 = CSH.UniformIntegerHyperparameter('L2', lower=100, upper=130, default_value=128, log=True)
            discount = CSH.UniformFloatHyperparameter('discount', lower=0.5, upper=1, default_value=0.9, log=False)
            DR = CSH.UniformFloatHyperparameter('decay', lower=0.00, upper=0.02, default_value=0.01, log=False)
            EP = CSH.UniformFloatHyperparameter('tradeoff', lower=0.50, upper=0.99, default_value=0.99, log=False)

            cs.add_hyperparameters([L1, L2, discount, DR, EP])

            # The hyperparameter sgd_momentum will be used,if the configuration
            # contains 'SGD' as optimizer.
            cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
            cs.add_condition(cond)
            return cs
        
sdir = 'RL_500 '
NS = hpns.NameServer(run_id='RL', host='127.0.0.1', port=None)
NS.start()
min_budget=10
max_budget=500
NWorker = 7
workers=[]
for i in range(NWorker):
    w = MCDQNWorker(nameserver='127.0.0.1', run_id='RL', id=i)
    w.run(background=True)
    workers.append(w)

# This example shows how to log live results. This is most useful
# for really long runs, where intermediate results could already be
# interesting. The core.result submodule contains the functionality to
# read the two generated files (results.json and configs.json) and
# create a Result object.
result_logger = hpres.json_result_logger(directory=sdir, overwrite=False)
bohb = BOHB(configspace=w.get_configspace(), run_id='RL', nameserver='127.0.0.1', result_logger=result_logger, min_budget=min_budget, max_budget= max_budget)
res = bohb.run(n_iterations=1, min_n_workers=NWorker)
# store results
with open(os.path.join(sdir, str('results_RL.pkl')), 'wb') as fh:
    pickle.dump(res, fh)

bohb.shutdown(shutdown_workers=True)
#rs = randomsearch.RandomSearch(w.config_space(), )
NS.shutdown()
