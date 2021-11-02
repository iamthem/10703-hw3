import numpy as np
import torch
from model_pytorch import make_model, ExpertModel
from utils import state_ndarray_to_tensor


def action_to_one_hot(env, action, batch):
    action_t = torch.tensor(action).long()
    return torch.clone(action_t).repeat(batch).reshape((batch))

# NOTE Some code borrowed form HW2 
def generate_episode(env, policy, batch):
    """Collects one rollout from the policy in an environment. The environment
    should implement the OpenAI Gym interface. A rollout ends when done=True. The
    number of states and actions should be the same, so you should not include
    the final state when done=True.

    Args:
    env: an OpenAI Gym environment.
    policy: The output of a deep neural network
    Returns:
    states: a list of states visited by the agent.
    actions: a list of actions taken by the agent. For tensorflow, it will be 
        helpful to use a one-hot encoding to represent discrete actions. The actions 
        that you return should be one-hot vectors (use action_to_one_hot()).
        For Pytorch, the Cross-Entropy Loss function will integers for action
        labels.
    rewards: the reward received by the agent at each step.
    """
    done = False
    state = env.reset()

    states = []
    rewards = []
    actions = []

    state_t = state_ndarray_to_tensor(state, batch) 
    new_state = None

    while not done:
            
        yhat = policy(state_t)
        # We Take best action during evaluation
        action = int(torch.argmax(yhat, dim=1)[0])
        new_state_np, reward, done, _ = env.step(action)
        del new_state
        new_state = state_ndarray_to_tensor(new_state_np, batch)
        states.append(new_state)
        actions.append(action_to_one_hot(env, action, batch))
        rewards.append(reward)
        state_t = new_state 

    env.close()

    return states, actions, rewards
        

class Imitation():
    
    def __init__(self, env, num_episodes, expert_file, device, batch = 4, nS = 4, nA = 2, expert_T = 200):
        self.env = env
        
        # Pytorch Only #
        self.expert = ExpertModel()
        self.expert.load_state_dict(torch.load(expert_file))
        self.expert.eval()
        self.expert_T = expert_T
        
        self.num_episodes = num_episodes
        
        self.nS = nS
        self.nA = nA
        self.batch = batch
        self.device = device
        self.model = make_model(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def generate_behavior_cloning_data(self):
        train_states = []
        train_actions = []

        for _ in range(self.num_episodes):
            states, y_hats, _ = generate_episode(self.env, self.expert, batch = self.batch)
            train_states.extend(states)
            train_actions.extend(y_hats)

        O_s = torch.cat(train_states).reshape(self.num_episodes, self.expert_T,
                                              self.batch, self.nS).to(self.device)
        O_a = torch.cat(train_actions).reshape(self.num_episodes, self.expert_T,
                                                 self.batch).to(self.device)

        return O_s, O_a
        
    def generate_dagger_data(self):
        # WRITE CODE HERE
        # You should collect states and actions from the student policy
        # (self.model), and then relabel the actions using the expert policy.
        # This method does not return anything.
        # END
        return
        
    def train(self, num_epochs=1, batch_size=64):
        """
        Train the model on data generated by the expert policy.
        Use Cross-Entropy Loss and a batch size of 64 when
        performing updates.
        Args:
            num_epochs: number of epochs to train on the data generated by the expert.
        Return:
            loss: (float) final loss of the trained policy.
            acc: (float) final accuracy of the trained policy
        """
        loss = 0
        acc = 0
        correct = 0

        for _ in range(num_epochs):

            O_s, O_a = self.generate_behavior_cloning_data()

            for episode in range(self.num_episodes):
                for t in range(self.expert_T):
                
                    state, y_batch = O_s[episode,t], O_a[episode, t]
                    yhat = self.model(state)
                    loss = self.criterion(yhat, y_batch)

                    # Backward prop.
                    self.optimizer.zero_grad()
                    loss.backward()
                    # Update model
                    self.optimizer.step()
                    
                    # Only compute correct labels  for final iteration
                    if episode == self.num_episodes - 1: 
                        correct += (torch.argmax(yhat[0], dim=0) == y_batch[0]).float().sum()

            acc = correct / (self.expert_T)

        return loss, acc


    def evaluate(self, policy, n_episodes=50):
        rewards = []
        for i in range(n_episodes):
            # We evaluate on 1 batch only
            _, _, r = generate_episode(self.env, policy, batch = 1)
            rewards.append(sum(r))
        r_mean = np.mean(rewards)
        return r_mean
    