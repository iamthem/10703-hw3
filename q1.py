import numpy as np
import matplotlib.pyplot as plt
import gym
import functools
from scipy.stats import multivariate_normal


def cmaes(fn, dim, num_iter=10):
  """Optimizes a given function using CMA-ES.
  Args:
    fn: A function that takes as input a vector and outputs a scalar value.
    dim: (int) The dimension of the vector that fn expects as input.
    num_iter: (int) Number of iterations to run CMA-ES.
  Returns:
    mu_vec: An array of size [num_iter, dim] storing the value of mu at each
      iteration.
    best_sample_vec: A list of length [num_iter] storing the function value
      for the best sample from each iteration of CMA-ES.
    mean_sample_vec: A list of length [num_iter] storing the average function
      value across samples from each iteration of CMA-ES.
  """
  # Hyperparameters
  sigma = 10
  population_size = 100
  p_keep = 0.10  # Fraction of population to keep
  noise = 0.25  # Noise added to covariance to prevent it from going to 0.
  keep = int(population_size * p_keep) # Number of survivors
  epsilon = noise * np.eye(dim) 

  # Initialize the mean and covariance
  mu = np.zeros(dim)
  cov = sigma**2 * np.eye(dim)
  
  mu_vec = []
  best_sample_vec = []
  mean_sample_vec = []

  mu_t = mu
  cov_t = cov 

  for _ in range(num_iter):

    Omega = multivariate_normal(mu_t, cov_t, allow_singular = True).rvs(population_size)
    scores = np.zeros(population_size)

    for i, param in enumerate(Omega):
        scores[i] = fn(param)

    Omega = Omega[scores.argsort()]
    survivors = Omega[-keep:]
    mu_t = 1/keep * np.sum(survivors, axis = 0)
    cov_t = np.cov(survivors.T) + epsilon

    mu_vec.append(mu_t)
    best_sample_vec.append(scores[-1])
    mean_sample_vec.append(scores.mean())
    del scores, Omega, survivors

  return mu_vec, best_sample_vec, mean_sample_vec

#In the cell below, we've defined a simply function:
#$$f(x) = -\|x - x^*\|_2^2 \quad \text{where} \quad x^* = [65, 49].$$
# This function is optimized when $x = x^*$. Run your implementation of CMA-ES on this function, confirming that you get the correct solution. 
# 

def test_fn(x):
  goal = np.array([65, 49])
  return -np.sum((x - goal)**2)



"""Run the following cell to visualize CMA-ES."""

def visualize(mu_vec):
    x = np.stack(np.meshgrid(np.linspace(-10, 100, 30), np.linspace(-10, 100, 30)), axis=-1)
    fn_value = [test_fn(xx) for xx in x.reshape((-1, 2))]
    fn_value = np.array(fn_value).reshape((30, 30))
    plt.figure(figsize=(6, 4))
    plt.contourf(x[:, :, 0], x[:, :, 1], fn_value, levels=10)
    plt.colorbar()
    mu_vec = np.array(mu_vec)
    plt.plot(mu_vec[:, 0], mu_vec[:, 1], 'b-o')
    init_l = list(np.round(mu_vec[0], 2))
    init_lab = 'initial value ' + str(init_l)
    plt.plot([mu_vec[0, 0]], [mu_vec[0, 1]], 'r+', ms=20, label=init_lab)
    fin_l = list(np.round(mu_vec[-1], 2))
    fin_lab = 'final value ' + str(fin_l)
    plt.plot([mu_vec[-1, 0]], [mu_vec[-1, 1]], 'g+', ms=20, label=fin_lab)
    plt.plot([65], [49], 'kx', ms=20, label='maximum ' + str([65,49]))
    plt.xlabel("mu_0")
    plt.ylabel("mu_1")
    plt.legend()
    plt.show()

#Next, you will apply CMA-ES to a more complicating: maximizing the expected reward of a RL agent. The policy takes action LEFT with probability:
# $$\pi(a = \text{LEFT} \mid s) = s \cdot w + b,$$
# where $w \in \mathbb{R}^4$ and $b \in \mathbb{R}$ are parameters that you will optimize with CMA-ES. In the cell below, define a function that takes as input a single vector $x = (w, b)$ and the environment and returns the total (undiscounted) reward from one episode.
# 

def _sigmoid(x):
  return 1 / (1 + np.exp(-x))

def _get_action(s, w, b):
  p_left = _sigmoid(w @ s + b)
  a = np.random.choice(2, p=[p_left, 1 - p_left])
  return a

def rl_fn(params, env, iters = 30):

    assert len(params) == 5
    rewards =  np.zeros(iters)
    policy = lambda s: _get_action(s, w, b)
    w = np.array(params[:4])
    b = np.array(params[4])

    for i in range(iters):
        state = env.reset()
        done = False
        total_rewards = 0

        while not done:
            action = policy(state)
            new_state, reward, done, _ = env.step(action) 
            state = new_state
            total_rewards += reward

        rewards[i] = total_rewards

    return rewards.mean()

"""The cell below applies your CMA-ES implementation to the RL objective you've defined in the cell above."""

def visualize_Cartpole(mu_vec, best_sample_vec, mean_sample_vec):
    plt.figure(figsize=(6, 4))
    mean_sample_reward = np.array(mean_sample_vec)
    best_sample_reward = np.array(best_sample_vec)
    iters = np.arange(1, 11)
    lab1 = "Mean Sample Reward" 
    lab2 = "Best Sample Reward" 
    plt.plot(iters, mean_sample_reward, label=lab1)
    plt.plot(iters, best_sample_reward, label=lab2)
    plt.xlabel("Iteration")
    plt.ylabel("Average Total Reward (over 30 Trajectories)")
    plt.hlines(195., 0, 10, linestyles = 'dashed', label = "Reward = 195")
    plt.legend(loc = "lower right")
    plt.show()

def cmaes_To_CartPole():
    env = gym.make('CartPole-v0')
    fn_with_env = functools.partial(rl_fn, env=env)
    mu_vec, best_sample_vec, mean_sample_vec = cmaes(fn_with_env, dim=5, num_iter=10)
    return (mu_vec, best_sample_vec, mean_sample_vec)
    visualize(mu_vec)


if __name__ == '__main__':
    mu_vec, best_sample_vec, mean_sample_vec = cmaes(test_fn, dim=2, num_iter=100)
    visualize(mu_vec)
