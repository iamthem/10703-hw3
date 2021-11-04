import torch
class ExpertModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4,24)
        self.fc2 = torch.nn.Linear(24,48)
        self.fc3 = torch.nn.Linear(48,2)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class FullyConnectedModel(torch.nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 10)
        self.fc2 = torch.nn.Linear(10, 2)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


def make_model(device, input_dim=4, output_dim=2):
    model = FullyConnectedModel(input_dim, output_dim)
    # We expect the model to have four weight variables (a kernel and bias for
    # both layers)
    assert len([p for p in model.parameters()]) == 4, 'Model should have 4 weights.'
    model.to(device)
    return model

