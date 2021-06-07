import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from ztw import InternalClassifier, SDNPool

NUM_THRESHOLDS = 150
START_LINSPACE = 0.1


class BigNatureCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        # original NatureCNN below
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class StandardInternalClassifier(InternalClassifier):
    def __init__(self, input_channels: int, inner_channels: int, output_dim: int, prev_dim: int = 0):
        super().__init__(input_channels, inner_channels, output_dim, prev_dim)
        self.conv1 = nn.Conv2d(input_channels, inner_channels, kernel_size=3, stride=2, padding=1)
        self.pool = SDNPool(inner_channels)
        if prev_dim > 0:
            self.stacking_norm = nn.LayerNorm(prev_dim)
        self.action_linear = nn.Linear(self.pool.after_pool_dim + prev_dim, output_dim)

    def forward(self, x: torch.Tensor, prev_output: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        if prev_output is not None:
            prev_output = self.stacking_norm(prev_output.detach())
            x = torch.cat([x, prev_output], -1)
        logits = self.action_linear(x)
        return logits


class BiggerInternalClassifier(InternalClassifier):
    def __init__(self, input_channels: int, inner_channels: int, output_dim: int, prev_dim: int = 0):
        super().__init__(input_channels, inner_channels, output_dim, prev_dim)
        self.conv1 = nn.Conv2d(input_channels, inner_channels, kernel_size=5, stride=1, padding=0)
        self.pool = SDNPool(inner_channels)
        if prev_dim > 0:
            self.stacking_norm = nn.LayerNorm(prev_dim)
        self.action_linear = nn.Linear(self.pool.after_pool_dim + prev_dim, output_dim)

    def forward(self, x: torch.Tensor, prev_output: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        if prev_output is not None:
            prev_output = self.stacking_norm(prev_output.detach())
            x = torch.cat([x, prev_output], -1)
        logits = self.action_linear(x)
        return logits


class SmallerInternalClassifier(InternalClassifier):
    def __init__(self, input_channels: int, inner_channels: int, output_dim: int, prev_dim: int = 0):
        super().__init__(input_channels, inner_channels, output_dim, prev_dim)
        self.conv1 = nn.Conv2d(input_channels, inner_channels, kernel_size=3, stride=4, padding=0)
        self.pool = SDNPool(inner_channels)
        if prev_dim > 0:
            self.stacking_norm = nn.LayerNorm(prev_dim)
        self.action_linear = nn.Linear(self.pool.after_pool_dim + prev_dim, output_dim)

    def forward(self, x: torch.Tensor, prev_output: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        if prev_output is not None:
            prev_output = self.stacking_norm(prev_output.detach())
            x = torch.cat([x, prev_output], -1)
        logits = self.action_linear(x)
        return logits
