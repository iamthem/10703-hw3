# %%
fn_with_env = functools.partial(q1.rl_fn, env=env)
fn = fn_with_env
dim = 5
sigma = 10
population_size = 100
p_keep = 0.10  # Fraction of population to keep
noise = 0.25  # Noise added to covariance to prevent it from going to 0.
epsilon = 0.25 * np.eye(dim) 
mu = np.zeros(dim)
cov = sigma**2 * np.eye(dim)
mu_t = mu
cov_t = cov 
keep = int(population_size * p_keep) # Number of survivors

# %%
cov_t.shape
Omega = multivariate_normal(mu_t, cov_t, allow_singular = True).rvs(population_size)
scores = np.zeros(population_size)

for i, param in enumerate(Omega):
    scores[i] = fn(param)

Omega = Omega[scores.argsort()]
survivors = Omega[-keep:]

# %%
mu_t = 1/keep * np.sum(survivors, axis = 0)
cov_t = np.array(
                    [np.cov(survivors[i], mu_t) for i in range(keep)]
                    ).mean(axis=0) 

survivors[0]
np.cov(survivors.T).shape
