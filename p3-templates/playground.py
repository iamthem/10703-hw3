# %%
from collections import OrderedDict 
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
import GCBC 
from importlib import reload
reload(GCBC)

# %%
# build env
l, T = 5, 30
env = GCBC.FourRooms(l, T)
GCBC.test_step(env,l)
