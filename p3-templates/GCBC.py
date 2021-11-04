from collections import OrderedDict 
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from model_pytorch import ExpertModel, make_model

### 3.1 Build Goal-Conditioned Task
class FourRooms:
    def __init__(self, l=5, T=30):
        '''
        FourRooms Environment for pedagogic purposes
        Each room is a l*l square gridworld, 
        connected by four narrow corridors,
        the center is at (l+1, l+1).
        There are two kinds of walls:
        - borders: x = 0 and 2*l+2 and y = 0 and 2*l+2 
        - central walls
        T: maximum horizion of one episode
        should be larger than O(4*l)
        '''
        assert l % 2 == 1 and l >= 5
        self.l = l
        self.total_l = 2 * l + 3
        self.T = T

        # create a map: zeros (walls) and ones (valid grids)
        self.map = np.ones((self.total_l, self.total_l), dtype=np.bool_)
        # build walls
        self.map[0, :] = self.map[-1, :] = self.map[:, 0] = self.map[:, -1] = False
        self.map[l+1, [1,2,-3,-2]] = self.map[[1,2,-3,-2], l+1] = False
        self.map[l+1, l+1] = False

        # define action mapping (go right/up/left/down, counter-clockwise)
        # e.g [1, 0] means + 1 in x coordinate, no change in y coordinate hence
        # hence resulting in moving right
        self.act_set = np.array([
        [1, 0], [0, 1], [-1, 0], [0, -1] 
        ], dtype=np.int_)
        self.action_space = spaces.Discrete(4)

        # you may use self.act_map in search algorithm 
        self.act_map = {}
        self.act_map[(1, 0)] = 0
        self.act_map[(0, 1)] = 1
        self.act_map[(-1, 0)] = 2
        self.act_map[(0, -1)] = 3

    def render_map(self):
        plt.imshow(self.map)
        plt.xlabel('y')
        plt.ylabel('x')
        plt.savefig('p2_map.png', 
                bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.show()

    def sample_sg(self):
        # sample s
        while True:
            s = [randint(self.total_l), randint(self.total_l)]
            if self.map[s[0], s[1]]:
                break

        # sample g
        while True:
            g = [randint(self.total_l), randint(self.total_l)]
            if self.map[g[0], g[1]] and (s[0] != g[0] or s[1] != g[1]):
                break
        return s, g

    def reset(self, s=None, g=None):
        '''
        s: starting position, np.array((2,))
        g: goal, np.array((2,))
        return obs: np.cat(s, g)
        '''
        if s is None or g is None:
            s, g = self.sample_sg()
        else:
            assert 0 < s[0] < self.total_l - 1 and 0 < s[1] < self.total_l - 1
            assert 0 < g[0] < self.total_l - 1 and 0 < g[1] < self.total_l - 1
            assert (s[0] != g[0] or s[1] != g[1])
            assert self.map[s[0], s[1]] and self.map[g[0], g[1]]

        self.s = s
        self.g = g
        self.t = 1

        return self._obs()

    def step(self, a):
        '''
        a: action, a scalar
        return obs, reward, done, info
        - done: whether the state has reached the goal
        - info: succ if the state has reached the goal, fail otherwise 
        '''
        assert self.action_space.contains(a)
        done = False
        info = () 

        # WRITE CODE HERE
        # END

        return self._obs(), 0.0, done, info

    def _obs(self):
        return np.concatenate([self.s, self.g])


def plot_traj(env, ax, traj, goal=None):
    traj_map = env.map.copy().astype(np.float32)
    traj_map[traj[:, 0], traj[:, 1]] = 2 # visited states
    traj_map[traj[0, 0], traj[0, 1]] = 1.5 # starting state
    traj_map[traj[-1, 0], traj[-1, 1]] = 2.5 # ending state
    if goal is not None:
        traj_map[goal[0], goal[1]] = 3 # goal
    ax.imshow(traj_map)
    ax.set_xlabel('y')
    ax.set_label('x')

### A uniformly random policy's trajectory
def test_step(env, l):
    s = np.array([1, 1])
    g = np.array([2*l+1, 2*l+1])
    s = env.reset(s, g)
    done = False
    traj = [s]
    while not done:
        s, _, done, _ = env.step(env.action_space.sample())
        traj.append(s)
    traj = np.array(traj)

    ax = plt.subplot()
    plot_traj(env, ax, traj, g)
    plt.savefig('p2_random_traj.png', 
        bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

# def shortest_path_expert(env):
# from queue import Queue
# N = 1000
# expert_trajs = []
# expert_actions = []

# # WRITE CODE HERE
# # END
# # You should obtain expert_trajs, expert_actions from search algorithm

# fig, axes = plt.subplots(5,5, figsize=(10,10))
# axes = axes.reshape(-1)
# for idx, ax in enumerate(axes):
# plot_traj(env, ax, expert_trajs[idx])

# plt.savefig('p2_expert_trajs.png', 
#     bbox_inches='tight', pad_inches=0.1, dpi=300)
# plt.show()

# def action_to_one_hot(env, action):
# action_vec = np.zeros(env.action_space.n)
# action_vec[action] = 1
# return action_vec  

# class GCBC:

# def __init__(self, env, expert_trajs, expert_actions):
# self.env = env
# self.expert_trajs = expert_trajs
# self.expert_actions = expert_actions
# self.transition_num = sum(map(len, expert_actions))
# self.model = make_model(input_dim=4, out_dim=4)
# # state_dim + goal_dim = 4
# # action_choices = 4

# def reset_model(self):
# self.model = make_model(input_dim=4, out_dim=4)	

# def generate_behavior_cloning_data(self):
# # 3 you will use action_to_one_hot() to convert scalar to vector
# # state should include goal
# self._train_states = []
# self._train_actions = []

# # WRITE CODE HERE
# # END

# self._train_states = np.array(self._train_states).astype(np.float) # size: (*, 4)
# self._train_actions = np.array(self._train_actions) # size: (*, 4)

# def generate_relabel_data(self):
# # 4 apply expert relabelling trick
# self._train_states = []
# self._train_actions = []

# # WRITE CODE HERE
# # END

# self._train_states = np.array(self._train_states).astype(np.float) # size: (*, 4)
# self._train_actions = np.array(self._train_actions) # size: (*, 4)

# def train(self, num_epochs=20, batch_size=256):
# """ 3
# Trains the model on training data generated by the expert policy.
# Args:
#     num_epochs: number of epochs to train on the data generated by the expert.
#     batch_size
# Return:
#     loss: (float) final loss of the trained policy.
#     acc: (float) final accuracy of the trained policy
# """
# # WRITE CODE HERE
# # END
# return loss, acc


# def evaluate_gc(env, policy, n_episodes=50):
# succs = 0
# for _ in range(n_episodes):
# info = generate_gc_episode(env, policy)
# # WRITE CODE HERE
# # END
# succs /= n_episodes
# return succs

# def generate_gc_episode(env, policy):
# """Collects one rollout from the policy in an environment. The environment
# should implement the OpenAI Gym interface. A rollout ends when done=True. The
# number of states and actions should be the same, so you should not include
# the final state when done=True.
# Args:
# env: an OpenAI Gym environment.
# policy: a trained model
# Returns:
# """
# done = False
# state = env.reset()
# while not done:
# # WRITE CODE HERE
# # END
# return info


# def run_GCBC(expert_trajs, expert_actions):
# gcbc = GCBC(env, expert_trajs, expert_actions)
# # mode = 'vanilla'
# mode = 'relabel'

# if mode == 'vanilla':
# gcbc.generate_behavior_cloning_data()
# else:
# gcbc.generate_relabel_data()

# print(gcbc._train_states.shape)

# num_seeds = 3
# loss_vecs = []
# acc_vecs = []
# succ_vecs = []

# for i in range(num_seeds):
# print('*' * 50)
# print('seed: %d' % i)
# loss_vec = []
# acc_vec = []
# succ_vec = []
# gcbc.reset_model()

# for e in range(200):
#     loss, acc = gcbc.train(num_epochs=20)
#     succ = evaluate_gc(env, gcbc)
#     loss_vec.append(loss)
#     acc_vec.append(acc)
#     succ_vec.append(succ)
#     print(e, round(loss,3), round(acc,3), succ)
# loss_vecs.append(loss_vec)
# acc_vecs.append(acc_vec)
# succ_vecs.append(succ_vec)

# loss_vec = np.mean(np.array(loss_vecs), axis = 0).tolist()
# acc_vec = np.mean(np.array(acc_vecs), axis = 0).tolist()
# succ_vec = np.mean(np.array(succ_vecs), axis = 0).tolist()

# ### Plot the results
# from scipy.ndimage import uniform_filter
# # you may use uniform_filter(succ_vec, 5) to smooth succ_vec
# plt.figure(figsize=(12, 3))
# # WRITE CODE HERE
# # END
# plt.savefig('p2_gcbc_%s.png' % mode, dpi=300)
# plt.show()

# def generate_random_trajs():
# N = 1000
# random_trajs = []
# random_actions = []
# random_goals = []

# # WRITE CODE HERE

# # END
# # You should obtain random_trajs, random_actions, random_goals from random policy

# # train GCBC based on the previous code
# # WRITE CODE HERE
