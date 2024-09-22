import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCnn(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv3d(
            1, 8, kernel_size=(20, 5, 5)
        )  # if (1, 201, 64, 64) --> (8, 182, 60, 60)
        self.pool1 = nn.MaxPool3d(
            kernel_size=1, stride=(4, 2, 2)
        )  # (8, 182, 60, 60) --> (8, 46, 30, 30)
        self.conv2 = nn.Conv3d(
            8, 16, kernel_size=(10, 5, 5)
        )  # (8, 46, 30, 30) --> (16, 37, 26, 26)
        self.pool2 = nn.MaxPool3d(
            kernel_size=2, stride=(2, 2, 2)
        )  # (16, 37, 26, 26) --> (16, 18, 13, 13)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(16 * 1 * 18 * 13 * 13, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
