# %%
import model_pytorch from torch.utils.data import DataLoader
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
batch = 32  
nS = 4
env = gym.make('CartPole-v0')
num_iterations = 50

# %%
reload(imitation)
reload(utils)
reload(model_pytorch)
im = imitation.Imitation(env, num_episodes, expert_file, device) 
O_s, O_a = im.generate_behavior_cloning_data()

# %%
Teacher_O_a = im.generate_dagger_data(O_s, O_a)
for i in range(im.expert_T):
    print(Teacher_O_a[0, i], O_a[ 0, i])
# %%
reload(utils)
torch.zeros(O_s.size()).shape
train_set = utils.Q2_Dataset(num_episodes * im.expert_T, batch, O_s, O_a)
train_loader = DataLoader(dataset=train_set) 
correct = 0
for i, (state_batch, action_batch) in enumerate(train_loader):
    y_hat = im.model(state_batch)
    loss = im.criterion(y_hat.squeeze(0), action_batch.squeeze(0))
    # Backward prop.
    im.optimizer.zero_grad()
    loss.backward()
    # Update model
    im.optimizer.step()
    if i % 200 == 0: 
        print("Step => ", i, " Loss at =>> ", loss)
    # if train_set.episode == num_episodes - 1: 
    #     correct +=    

# %%
for _ in range(num_iterations):
    loss, acc, reward = im.train(batch_size = batch)
    print("Loss => ", loss, "Reward ==> ", reward)
