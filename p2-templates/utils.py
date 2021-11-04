import torch
from torch.utils.data import Dataset

def state_ndarray_to_tensor(state_array, batch):
    return torch.clone(torch.from_numpy(state_array).float()).repeat(batch).reshape((batch, state_array.shape[0]))


class Q2_Dataset(Dataset):
    def __init__(self, num_episodes, batch, D, nS, mode, expert_T):
        self.num_episodes = num_episodes 
        self.batch = batch
        self.D = D
        self.d = 0
        self.nS = nS
        self.device = 'cpu' 
        self.mode = mode
        self.expert_T = expert_T

    def __len__(self): 
        return len(self.D)

    def __getitem__(self, idx):

        assert self.__len__ () > 0 and self.d < self.__len__()
        states, actions = self.D[self.d]
        
        if self.mode == 'dagger':

            if self.batch > states.shape[1]:
                rep = self.batch // states.shape[1]
                batch = rep * states.shape[1]
            else:
                batch = self.batch

            O_s = torch.zeros((self.num_episodes, batch, self.nS), device = self.device).float()
            O_a = torch.zeros((self.num_episodes, batch), device = self.device).long()

            for episode in range(self.num_episodes):
                # Sample episodes in random order 
                indices = torch.randperm(states.shape[1])[:self.batch]

                if self.batch > states.shape[1]:
                    indices = indices.repeat(rep)

                assert(O_s[episode].shape == states[episode, indices].shape)
                O_s[episode] = states[episode, indices]
                O_a[episode] = actions[episode, indices]

        else: 
            O_s = torch.zeros((self.num_episodes, self.expert_T, self.batch, self.nS), device = self.device).float()
            O_a = torch.zeros((self.num_episodes, self.expert_T, self.batch), device = self.device).long()

            for episode in range(self.num_episodes):
                for t in range(self.expert_T):
                    # Sample batches of trajectory 
                    indices = torch.randperm(self.expert_T)[:self.batch]
                    O_s[episode, t] = states[episode, indices]
                    O_a[episode, t] = actions[episode, indices]

        self.d += 1
        return O_s, O_a 

class Yikes(Exception): pass 
