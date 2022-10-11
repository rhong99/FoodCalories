import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, input_size, first_out, second_out, third_out, out):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(input_size, first_out)
        self.fc2 = nn.Linear(first_out, second_out)
        self.fc3 = nn.Linear(second_out, third_out)
        self.fc4 = nn.Linear(third_out, out)

    def forward(self, features):
        x = torch.flatten(features, start_dim=1, end_dim=3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return torch.sigmoid(x)
