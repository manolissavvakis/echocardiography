import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCnn(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv3d(
            1, 8, kernel_size=(5, 5, 5)
        )  # if 1 x depth=20 x 256 x 256 --> 8 x 16 x 252 x 252
        self.pool1 = nn.MaxPool3d(
            kernel_size=1, stride=(4, 2, 2)
        )  # 8 x 16 x 252 x 252 --> 8 x 4 x 126 x 126
        self.conv2 = nn.Conv3d(
            8, 16, kernel_size=(3, 5, 5)
        )  # 8 x 4 x 126 x 126 --> 16 x 2 x 122 x 122
        self.pool2 = nn.MaxPool3d(
            kernel_size=2, stride=(1, 2, 2)
        )  # 16 x 2 x 122 x 122 --> 16 x 1 x 61 x 61
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(16 * 1 * 61 * 61, 64)
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
