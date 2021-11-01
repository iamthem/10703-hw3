# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import functools

# %%
sigma = 10
population_size = 100
p_keep = 0.10  # Fraction of population to keep
noise = 0.25  # Noise added to covariance to prevent it from going to 0.
dim = 2
def test_fn(x):
  goal = np.array([65, 49])
  return -np.sum((x - goal)**2)

epsilon = 0.25 * np.eye(dim) 

# %%
mu = np.zeros(dim)
cov = sigma**2 * np.eye(dim)
Omega = multivariate_normal(mu, cov).rvs(population_size)
scores = np.zeros(Omega.shape[0])

for i, param in enumerate(Omega):
    scores[i] = test_fn(param)

# %%
Omega = Omega[scores.argsort()]
keep = int(population_size * p_keep)
survivors = Omega[-keep:]
survivors.shape

# %%
mu_t = 1/keep * np.sum(survivors, axis = 0)
A = np.array([np.cov(survivors[i], mu_t) for i in range(survivors.shape[0])]).mean(axis=0)

# %%
