import torch
import torch.nn as nn
from torch.nn import init
import numpy as np


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class RNDNetwork(nn.Module):
    def __init__(self):
        super(RNDNetwork, self).__init__()

        feature_output = 4 * 4 * 64
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=8,
                      stride=4), nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                      stride=2), nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                      stride=1), nn.LeakyReLU(), Flatten(),
            nn.Linear(feature_output, 256), nn.ReLU(), nn.Linear(256, 256),
            nn.ReLU(), nn.Linear(256, 256))

        self.target = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=8,
                      stride=4), nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                      stride=2), nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                      stride=1), nn.LeakyReLU(), Flatten(),
            nn.Linear(feature_output, 256))

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        #print(next_obs)
        next_obs_ = next_obs["tactile_image"].float() / 255.0
        target_feature = self.target(next_obs_)
        predict_feature = self.predictor(next_obs_)

        return predict_feature, target_feature
