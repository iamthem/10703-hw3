# %%
import model_pytorch
from importlib import reload
import imitation
import model_pytorch
import gym
import numpy as np
import torch
import utils
num_episodes = 100
expert_file = 'expert_torch.pt'
device = 'cpu'
batch = 16 
nS = 4
env = gym.make('CartPole-v0')
num_iterations = 30

# %%
reload(utils)
reload(imitation)
reload(model_pytorch)
im = imitation.Imitation(env, num_episodes, expert_file, device, batch = batch)
for i in range(num_iterations):
    loss, acc, reward = im.train(batch_size = batch)
    print("Loss => ", loss,
          "\nAccuracy => ", acc,
          "Reward ==> ", reward)
