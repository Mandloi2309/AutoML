# AutoML
This repository deals with HPO of a Reinforcement problem - Mountain Car

The script AML-MC.py uses OpenAI Gym environments for Mountain car.
  Here we use a nerual network for solving the mountain car problem.
  In order to learn the best policy we use a reinforcemnet learning concept called Q-learning, which improves the agent's action based on the reward recieved.
  
  This model has numerous hyperparameters which can be optimised and therefore we used BOHB implemented by Stefan Falkner, Aaron Klein and Frank Hutter (arXiv:1807.01774)
  We optimized a total of 9 hyperparameters. 
