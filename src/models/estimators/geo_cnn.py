import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class GCNNWR(nn.Module):
    def __init__(self, dim, in_channels=1):
        super(GCNNWR, self).__init__()
        self.dim = dim
        init = 0.1

        self.conv_layers = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels, 32, kernel_size=3, padding='same')),
            nn.PReLU(init),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            weight_norm(nn.Conv2d(32, 64, kernel_size=3, padding='same')),
            nn.PReLU(init),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            weight_norm(nn.Conv2d(64, 256, kernel_size=3, padding='same')),
            nn.PReLU(init),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(init),
            
            nn.Linear(128, 64),
            nn.PReLU(init),
            
            nn.Linear(64, 16),
            nn.PReLU(init),
            
            nn.Linear(16, self.dim+1)
        )

        self.beta_ols = torch.ones(self.dim+1).float()

    def forward(self, inputs_dis, inputs_x):
        x = self.conv_layers(inputs_dis)
        x = self.fc_layers(x)

        y_pred = x[:, 0]*self.beta_ols[0]
        for w in range(self.dim-1):
            y_pred += x[:, w+1]*inputs_x[:, w]*self.beta_ols[w+1]

        return y_pred, x