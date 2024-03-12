import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCnn(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 7, kernel_size=(5, 5, 5)) # if 1 x depth=20 x 256 x 256 --> 7 x 16 x 252 x 252
        self.conv2 = nn.Conv3d(7, 17, kernel_size=(3, 5, 5)) # 7 x 4 x 126 x 126 --> 17 x 2 x 122 x 122
        self.pool1 = nn.MaxPool3d(kernel_size=1, stride=(4, 2, 2)) # 7 x 16 x 252 x 252 --> 7 x 4 x 126 x 126
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=(1, 1, 1)) # 17 x 2 x 122 x 122 --> 17 x 1 x 121 x 121
        self.fc1 = nn.Linear(17 * 1 * 121 * 121, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x
