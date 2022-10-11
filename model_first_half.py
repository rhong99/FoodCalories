import torch
import torch.nn as nn
import torch.nn.functional as F


class model_1(nn.Module):
    def __init__(self, k1, k2, k3, k4, l1, l2, outputs):
        super(model_1, self).__init__()
        self.conv1 = nn.Conv2d(3, k1, 9)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(k1, k2, 5)
        self.conv3 = nn.Conv2d(k2, k3, 3)
        self.conv4 = nn.Conv2d(k3, k4, 3)
        self.fc1 = nn.Linear(k4*5*5, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, outputs)

    def forward(self, features):
        x = self.pool(F.relu(self.conv1(features)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
