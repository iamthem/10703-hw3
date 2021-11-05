# %%
from collections import OrderedDict 
import torch
from model_pytorch import ExpertModel
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
import GCBC 
from importlib import reload
reload(GCBC)
import heapq
l, T = 5, 30
env = GCBC.FourRooms(l, T)
init = env.reset()
from itertools import chain

# %%
et, ea = GCBC.generate_expert_trajectories(env, 5)
np.pad(et[0], [(0,1), (0,0)], mode = 'constant', constant_values = -1)

