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
mode = 'dagger'
expert_T = 200
D = list() 
reload(utils)
reload(BCDAGGER)
reload(imitation)
reload(model_pytorch)
minibatch = 2
im = imitation.Imitation(env, num_episodes, expert_file, device, mode, batch, 200, minibatch)
keys = [num_episodes]
num_seeds = 1

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
    print(loss)

# %%
train_set = utils.Q2_Dataset(num_episodes, batch, D, nS, device) 
train_loader = DataLoader(dataset=train_set) 
O_s_og, Teacher_O_a = im.generate_dagger_data()
D.append((O_s_og, Teacher_O_a))
O_s, O_a = next(iter(train_loader))

# %%
state_batch = O_s[0,0]
action_batch = O_a[0,0]
indices = torch.randperm(state_batch.shape[0])[:batch]
y_hat = im.model(state_batch[indices])

# %%
train_states = []
train_actions = []
episode_lens = []
for _ in range(num_episodes):
    states, y_hats, rewards = imitation.generate_episode(env, im.model)
    train_states.extend(states)
    train_actions.extend(y_hats)
    episode_lens.append(sum(rewards))
    
episode_lens_t = torch.tensor(episode_lens).long().to(device)
O_s = torch.cat(train_states).float().to(device)
O_a = torch.cat(train_actions).long().to(device)


# %%
D.append((O_s, O_a))
train_set = utils.Q2_Dataset(num_episodes, expert_T, batch, D, nS, device) 
train_loader = DataLoader(dataset=train_set) 

# %%
count = 0
for d, (O_s, O_a) in enumerate(train_loader):
    for episode in range(num_episodes):
        for t in range(expert_T):
            count += 1
            assert O_s[0, episode, t].shape == torch.Size([batch, nS]) 
            y_hat = im.model(O_s[0, episode, t])
            loss = im.criterion(y_hat.squeeze(0), O_a[0, episode, t].squeeze(0))
            assert loss.shape == torch.Size([]) 

assert count == len(D) * num_episodes * expert_T
# %%
Teacher_O_a = im.generate_dagger_data(O_s, O_a)
res = torch.zeros((len(D), num_episodes, 200, 4))
indices = torch.randperm(expert_T)[:batch]
O_s[0, indices].shape

# %%
countEquals = 0
for i in range(im.expert_T): 
    if Teacher_O_a[0, i] == O_a[ 0, i]:
        countEquals += 1
    print(Teacher_O_a[0, i], O_a[ 0, i])
countEquals

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

