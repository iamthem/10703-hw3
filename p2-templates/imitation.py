import numpy as np
import torch
from torch.utils.data import DataLoader
from model_pytorch import make_model, ExpertModel
from utils import state_ndarray_to_tensor, Q2_Dataset 


def action_to_one_hot(action, batch):
    action_t = torch.tensor(action).long()
    return torch.clone(action_t).repeat(batch).reshape((batch))

# NOTE Some code borrowed form HW2 
def generate_episode(env, policy, device):
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

    state_t = state_ndarray_to_tensor(state, batch = 1).to(device)
    new_state = None

    while not done:
            
        yhat = policy(state_t)
        # We Take best action during evaluation
        action = int(torch.argmax(yhat, dim=1)[0])
        new_state_np, reward, done, _ = env.step(action)
        del new_state
        new_state = state_ndarray_to_tensor(new_state_np, batch = 1).to(device)
        states.append(new_state)
        actions.append(action_to_one_hot(action, batch = 1).to(device))
        rewards.append(reward)
        state_t = new_state 

    env.close()

    return states, actions, rewards
        

class Imitation():
    
    def __init__(self, env, num_episodes, expert_file, device, mode, batch = 64, expert_T = 200):
        self.env = env
        
        # Pytorch Only #
        self.expert = ExpertModel().to(device)
        self.expert.load_state_dict(torch.load(expert_file))
        self.expert.eval()
        self.expert_T = expert_T
        self.mode = mode
        self.batch = batch
        
        self.num_episodes = num_episodes
        
        self.nS = env.observation_space.shape[0]
        self.nA = env.action_space.n
        self.device = device
        self.model = make_model(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.total_loss = 0
        self.acc_total = 0
        self.training_iters = 0
        
    def tensor_trajectory(self, train_states, train_actions, episode_lens):
        shortest_episode = int(np.min(np.array(episode_lens)))

        O_s = torch.zeros((self.num_episodes, shortest_episode, self.nS)).float().to(self.device)
        O_a = torch.zeros((self.num_episodes, shortest_episode)).long().to(self.device)

        start = 0
        for e in range(self.num_episodes):
            if e > 0:
                start = start + episode_lens[e-1] 

            end = start + shortest_episode 
            assert end - start == shortest_episode
            O_s[e, :] = torch.cat(train_states[ start : end ] , 1).reshape(shortest_episode, self.nS)
            O_a[e, :] = torch.cat(train_actions[ start : end ], 0)

        return O_s, O_a
        
    def generate_behavior_cloning_data(self, expert = True):
        train_states = []
        train_actions = []
        episode_lens = []

        for _ in range(self.num_episodes):

            if expert: 
                states, actions, rewards = generate_episode(self.env, self.expert, self.device)
            else: 
                states, actions, rewards = generate_episode(self.env, self.model, self.device)

            train_states.extend(states)
            train_actions.extend(actions)
            episode_lens.append(len(rewards))
        
        return self.tensor_trajectory(train_states, train_actions, episode_lens)
        

    def generate_dagger_data(self):
        O_s, O_a = self.generate_behavior_cloning_data(expert=False)

        Teacher_O_a = torch.zeros(O_a.size()).to(self.device)

        for episode in range(self.num_episodes):

            # For length of current episode
            for t in range(O_a.shape[1]):

                yhat = self.expert(O_s[episode, t])
                action = int(torch.argmax(yhat, dim=0))
                Teacher_O_a[episode, t] = action_to_one_hot(action, batch = 1).to(self.device)

        return O_s, Teacher_O_a 

    def step(self, state_batch, action_batch):
        y_hat = self.model(state_batch)
        loss = self.criterion(y_hat.squeeze(0), action_batch.squeeze(0))
        
        # Backward prop.
        self.optimizer.zero_grad()
        loss.backward()

        # Update model
        self.optimizer.step()

        # Add to loss, acc 
        self.total_loss += loss
        #TODO update acc_totals
        self.acc_total += 0.5
        self.training_iters += 1


    def Learn(self, train_loader):
        for _, (O_s, O_a) in enumerate(train_loader):
            for episode in range(O_a[0].shape[0]):
                for t in range(O_a[0].shape[1]):
                    state_batch = O_s[0, episode, t]
                    action_batch = O_a[0, episode, t]
                    self.step(state_batch, action_batch)
    
    def train(self, D = list()):
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
        self.total_loss = 0
        self.acc_total = 0
        self.training_iters = 0
        train_set = Q2_Dataset(self.num_episodes, self.batch, D, self.nS, self.device) 
        train_loader = DataLoader(dataset=train_set) 

        if self.mode == 'behavior cloning':
            O_s, O_a = self.generate_behavior_cloning_data()
            D.append((O_s, O_a))
            self.Learn(train_loader)

        if self.mode == 'dagger':
            O_s, Teacher_O_a = self.generate_dagger_data() 
            D.append((O_s, Teacher_O_a))
            self.Learn(train_loader)
            
        acc_mean = self.acc_total / self.training_iters 
        loss_mean = self.total_loss / self.training_iters 

        return loss_mean, acc_mean, D


    def evaluate(self, policy, n_episodes=50):
        rewards = []
        for _ in range(n_episodes):
            # We evaluate on 1 batch only
            _, _, r = generate_episode(self.env, policy, self.device)
            rewards.append(sum(r))
        r_mean = np.mean(rewards)
        return r_mean
