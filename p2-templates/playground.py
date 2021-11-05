# %%
from scipy.ndimage.filters import uniform_filter1d
import model_pytorch
from torch.utils.data import DataLoader
from importlib import reload
import imitation
import model_pytorch
import gym
import numpy as np
import torch
import BCDAGGER
import utils
num_episodes = 15 
expert_file = 'expert_torch.pt'
device = 'cpu'
batch = 8   
nS = 4
nA = 2
env = gym.make('CartPole-v0')
num_iterations = 5 
mode = 'behavior cloning'
expert_T = 200
D = list() 
reload(utils)
reload(BCDAGGER)
reload(imitation)
reload(model_pytorch)
minibatch = 2
im = imitation.Imitation(env, num_episodes, expert_file, device, mode, batch, 200, minibatch)
keys = [1,2, 5]
num_seeds = 2
BCDAGGER.plot_compare_num_episodes(mode, expert_file, device, keys, num_seeds=num_seeds, num_iterations=num_iterations)





# %%
BCDAGGER.plot_student_vs_expert(mode, expert_file, device, keys, num_seeds, num_iterations)


# %%
loss_vec = np.zeros(num_iterations) 
acc_vec = np.zeros(num_iterations) 
imitation_reward_vec = np.zeros(num_iterations) 

for i in range(num_iterations):
    loss, acc, D = im.train(D)
    loss_vec[i] = loss
    acc_vec[i] = acc
    imitation_reward_vec[i] = im.evaluate(im.model)
    print(imitation_reward_vec[i])

# %%
train_set = utils.Q2_Dataset(num_episodes, batch, D, nS, device) 
train_loader = DataLoader(dataset=train_set) 
O_s_og, Teacher_O_a = im.generate_dagger_data()
D.append((O_s_og, Teacher_O_a))
O_s, O_a = next(iter(train_loader))

