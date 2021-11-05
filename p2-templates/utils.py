import torch
from torch.utils.data import Dataset

def state_ndarray_to_tensor(state_array, batch):
    return torch.clone(torch.from_numpy(state_array).float()).repeat(batch).reshape((batch, state_array.shape[0]))

def batch_idx(batch, sample_size):
    
    # Shuffle samples 
    indices = torch.randperm(sample_size) 
    new_batch = batch
    
    # If batch is greater than episode length repeat episode batch // num_episodes times
    # And caculate new batch size since batch % sample != 0 sometimes
    if batch > sample_size:
        rep = batch // sample_size 
        new_batch = rep * sample_size 
        indices = indices.repeat(rep)

    else:
        indices = indices[:batch]

    return new_batch, indices

class Q2_Dataset(Dataset):
    def __init__(self, num_episodes, batch, D, nS, device):
        self.num_episodes = num_episodes 
        self.batch = batch
        self.D = D
        self.d = 0
        self.nS = nS
        self.device = device

    def __len__(self): 
        return len(self.D)

    def __getitem__(self, idx):

        assert self.__len__ () > 0 and self.d < self.__len__()
        states, actions = self.D[self.d]
        
        # Random permutation of episodes 
        batch, indices = batch_idx(self.batch, self.num_episodes) 
        
        # (batch, shortest_episode, state.size ) 
        O_s = torch.zeros((batch, states.shape[1], self.nS)).float().to(self.device)
        # (batch, shortest_episode ) 
        O_a = torch.zeros((batch, states.shape[1])).long().to(self.device)

        O_s = states[indices]
        O_a = actions[indices] 

        self.d += 1
        return O_s, O_a 

class Yikes(Exception): pass 
