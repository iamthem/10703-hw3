import torch
from torch.utils.data import Dataset

def state_ndarray_to_tensor(state_array, batch):
    return torch.clone(torch.from_numpy(state_array).float()).repeat(batch).reshape((batch, state_array.shape[0]))


class Q2_Dataset(Dataset):
    def __init__(self, iters, batch, O_s, O_a):
        self.length = iters 
        self.batch = batch
        self.t = 0
        self.episode = 0
        self.O_s = O_s
        self.O_a = O_a

    def __len__(self): 
        return self.length

    def __getitem__(self, idx):

        assert self.episode < self.O_a.shape[0]
        assert self.t < self.O_a.shape[1]

        idx = torch.randperm(self.O_a[self.episode].shape[0])[:self.batch]

        states = self.O_s[self.episode, idx]
        actions = self.O_a[self.episode, idx]
        self.t += 1

        # NOTE We are assuming each episode from expert has the same T
        if self.t % self.O_a.shape[1] == 0:
            self.t = 0
            self.episode += 1

        return states, actions

