#Main file for Assignment 5
from my_env import *
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as mpl
import sklearn.preprocessing as skl


def main():
    #Hoi Laura, alle classes staan in my_env

    n_iter = 1000  # Number of iterations
    env = EvidenceEnv(n=2,p=0.95) # environment specs
    agent = RandomAgent(env) # define agent
    obs = env.reset() # reset environment and agent
    reward = None
    done = False
    R = []
    for step in range(n_iter):
        env.render()
        action = agent.act(obs)
        _obs, reward, done, _ = env.step(action)
        # no training involved for random agent
        agent.train(action, obs, reward, _obs)
        obs = _obs
        R.append(reward)

    #1: Run the code and plot the cumulative reward over time
    cu_reward = [0]
    for i in range(1,1000):
        cu_reward.append(cu_reward[i-1] + R[i]) #Calculate cumulative reward for each timestep
    mpl.plot(range(0,n_iter),cu_reward)
    mpl.show()

    #2: Implement  the  Tabular  Q-learning  algorithm  and  show  that  your  TabularQAgent  learns  to
    # accumulate reward over time. Plot the cumulative rewards for this agent. Also plot the values for
    # Q(s,a) before and after learning.



if(__name__ == "__main__"):
    main()