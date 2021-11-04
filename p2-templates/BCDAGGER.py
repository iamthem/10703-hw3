"""F21 10-703 HW3
# 10-703: Homework 3 Question 2 - Behavior Cloning & DAGGER

You will implement this assignment in this python file

You are given helper functions to plot all the required graphs
"""

from collections import OrderedDict 
import gym
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.ndimage.filters import uniform_filter1d

from imitation import Imitation
from tqdm import tqdm
	

def generate_imitation_results(mode, expert_file, device, keys=[100], num_seeds=1, num_iterations=100, batch_size = 64):
    # Use a small number of training iterations
    # (e.g., 10) for debugging, and then try a larger number
    # (e.g., 100).

    # Dictionary mapping number of expert trajectories to a list of rewards.
    # Each is the result of running with a different random seed.
    # At the end of the function:
    # 	reward_data is a dictionary with keys for each # expert trajectories
    #	reward_data[i] will be a list of length num_seeds containing lists 
    #	reward_data[i][j] will be a list of length num_iterations containing
    #	rewards each iteration using key i and seed j.
    reward_data = OrderedDict({key: [] for key in keys})
    accuracy_data = OrderedDict({key: [] for key in keys})
    loss_data = OrderedDict({key: [] for key in keys})
    expert_reward = None 

    for num_episodes in keys:
        for t in range(num_seeds):
            print('*' * 50)
            print('num_episodes: %s; seed: %d' % (num_episodes, t))

            # Create the environment.
            env = gym.make('CartPole-v0')
            env.seed(t) # set seed
            im = Imitation(env, num_episodes, expert_file, device, mode, batch = batch_size, expert_T=200, minibatch=8)
            expert_reward = im.evaluate(im.expert)

            loss_vec = np.zeros(num_iterations) 
            acc_vec = np.zeros(num_iterations) 
            imitation_reward_vec = np.zeros(num_iterations) 
            D = list() 

            for i in tqdm(range(num_iterations)):
                loss, acc, D = im.train(D)
                loss_vec[i] = loss
                imitation_reward_vec[i] = im.evaluate(im.model)
                acc_vec[i] = acc

            reward_data[num_episodes].append(uniform_filter1d(imitation_reward_vec, size=num_iterations))
            accuracy_data[num_episodes].append(uniform_filter1d(acc_vec, size=num_iterations)) 
            loss_data[num_episodes].append(uniform_filter1d(loss_vec, size=num_iterations))
        
    return reward_data, accuracy_data, loss_data, expert_reward


# """### Experiment: Student vs Expert
# In the next two cells, you will compare the performance of the expert policy
# to the imitation policies obtained via behavior cloning and DAGGER.
# """
def plot_student_vs_expert(mode, expert_file, device, keys=[100], num_seeds=1, num_iterations=100):
    assert len(keys) == 1
    reward_data, acc_data, loss_data, expert_reward = \
    generate_imitation_results(mode, expert_file, device, keys, num_seeds, num_iterations)

    for key in keys:
        reward_arr = np.array(reward_data[key][0])
        acc_arr = np.array(acc_data[key][0])
        loss_arr = np.array(loss_data[key][0])
    x = np.arange(1, num_iterations+1)
    expert_reward_array = np.array([expert_reward] * num_iterations)

    # Plot the results
    plt.figure(figsize=(12, 3))
    fig, axarr = plt.subplots(3,1)
    
    axarr[0].plot(x, reward_arr)
    axarr[0].plot(x, expert_reward_array, linestyle = 'dashed', label = 'Expert Reward')
    axarr[0].legend()
    axarr[1].plot(x, acc_arr)
    axarr[2].plot(x, loss_arr)
    plt.setp(axarr[0], ylabel='Reward')
    plt.setp(axarr[1], ylabel='Accuracy')
    plt.setp(axarr[2], ylabel='Loss')
    plt.xlabel("Iterations")

    # END
    plt.savefig('q2-2-1.png', dpi=300)
    #plt.show()

# """Plot the reward, loss, and accuracy for each, remembering to label each line."""
# def plot_compare_num_episodes(mode, expert_file, device, keys, num_seeds=1, num_iterations=100):
# s0 = time.time()
# reward_data, accuracy_data, loss_data, _ = \
#     generate_imitation_results(mode, expert_file, device, keys, num_seeds, num_iterations)

# # Plot the results
# plt.figure(figsize=(12, 4))
# # WRITE CODE HERE

# # END
# plt.savefig('p1_expert_data_%s.png' % mode, dpi=300)
# # plt.show()
# print('time cost', time.time() - s0)


def main():
	# Generate all plots for Problem 1
	
    expert_file = 'expert_torch.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Switch mode
    #mode = 'behavior cloning'
    mode = 'dagger'

    # Change the list of num_episodes below for testing and different tasks
    keys = [100] # [1, 10, 50, 100]
    num_seeds = 1 # 3
    num_iterations = 100    # Number of training iterations. Use a small number
                            # (e.g., 10) for debugging, and then try a larger number
                            # (e.g., 100).

    # Q2.1.1, Q2.2.1
    plot_student_vs_expert(mode, expert_file, device, keys, num_seeds=num_seeds, num_iterations=num_iterations)

    # # Q2.1.2, Q2.2.2
    # plot_compare_num_episodes(mode, expert_file, device, keys, num_seeds=num_seeds, num_iterations=num_iterations)


if __name__ == '__main__':
	main()

