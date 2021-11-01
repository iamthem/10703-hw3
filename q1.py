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
  epsilon = 0.25 * np.eye(dim) 

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
    scores = np.zeros(Omega.shape[0])

    for i, param in enumerate(Omega):
        scores[i] = fn(param)

    Omega = Omega[scores.argsort()]
    survivors = Omega[-keep:]
    mu_t = 1/keep * np.sum(survivors, axis = 0)
    cov_t = np.array(
                    [np.cov(survivors[i], mu_t) for i in range(keep)]
                    ).mean(axis=0)

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
    plt.plot([mu_vec[0, 0]], [mu_vec[0, 1]], 'r+', ms=20, label='initial value')
    plt.plot([mu_vec[-1, 0]], [mu_vec[-1, 1]], 'g+', ms=20, label='final value')
    plt.plot([65], [49], 'kx', ms=20, label='maximum')
    plt.legend()
    plt.show()

#Next, you will apply CMA-ES to a more complicating: maximizing the expected reward of a RL agent. The policy takes action LEFT with probability:
# $$\pi(a = \text{LEFT} \mid s) = s \cdot w + b,$$
# where $w \in \mathbb{R}^4$ and $b \in \mathbb{R}$ are parameters that you will optimize with CMA-ES. In the cell below, define a function that takes as input a single vector $x = (w, b)$ and the environment and returns the total (undiscounted) reward from one episode.
# 

def _sigmoid(x):
  return 1 / (1 + np.exp(-x))

def _get_action(s, params):
  w = params[:4]
  b = params[4]
  p_left = _sigmoid(w @ s + b)
  a = np.random.choice(2, p=[p_left, 1 - p_left])
  return a

def rl_fn(params, env):
  assert len(params) == 5
  ## calculate total_rewards
  pass 

"""The cell below applies your CMA-ES implementation to the RL objective you've defined in the cell above."""

def cmaes_To_CartPole():
    env = gym.make('CartPole-v0')
    fn_with_env = functools.partial(rl_fn, env=env)
    mu_vec, best_sample_vec, mean_sample_vec = cmaes(fn_with_env, dim=5, num_iter=10)


if __name__ == '__main__':
    mu_vec, best_sample_vec, mean_sample_vec = cmaes(test_fn, dim=2, num_iter=100)    
    print(mu_vec[-1]) 
