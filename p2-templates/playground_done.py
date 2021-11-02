# %%
def _sigmoid(x):
  return 1 / (1 + np.exp(-x))

def _get_action(s, w, b):
  p_left = _sigmoid(w @ s + b)
  a = np.random.choice(2, p=[p_left, 1 - p_left])
  return a

reload(imitation)
w = np.array([-1,-1,-1,-1])
b = np.array(-1)
policy = lambda s: _get_action(s, w, b)
total_rewards =  np.zeros(1000)

for i in range(1000):
    states, actions, rewards = imitation.generate_episode(env, policy)
    total_rewards[i] = np.sum(rewards)

np.mean(total_rewards)


# Q2 BC Part 1
# %%
O_s, O_a = im.generate_behavior_cloning_data()
iters = 10
correct = 0
for episode in range(iters):
    for t in range(im.expert_T):
    
        state, y_batch = O_s[episode,t], O_a[episode, t]
        yhat = im.model(state)
        loss = im.criterion(yhat, y_batch)

        # Backward prop.
        im.optimizer.zero_grad()
        loss.backward()
        # Update model
        im.optimizer.step()

        correct += (torch.argmax(yhat[0], dim=0) == y_batch[0]).float().sum()

acc = correct / (iters * im.expert_T)
acc
