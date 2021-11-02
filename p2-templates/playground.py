# %%
import model_pytorch
from importlib import reload
import imitation
import gym
import numpy as np
env = gym.make('CartPole-v0')

def _sigmoid(x):
  return 1 / (1 + np.exp(-x))

def _get_action(s, w, b):
  p_left = _sigmoid(w @ s + b)
  a = np.random.choice(2, p=[p_left, 1 - p_left])
  return a

# %%
reload(imitation)
w = np.array([-1,-1,-1,-1])
b = np.array(-1)
policy = lambda s: _get_action(s, w, b)
total_rewards =  np.zeros(1000)

for i in range(1000):
    states, actions, rewards = imitation.generate_episode(env, policy)
    total_rewards[i] = np.sum(rewards)

np.mean(total_rewards)

