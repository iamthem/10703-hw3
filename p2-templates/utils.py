import torch

def state_ndarray_to_tensor(state_array, batch):
    return torch.clone(torch.from_numpy(state_array).float()).repeat(batch).reshape((batch, state_array.shape[0]))
