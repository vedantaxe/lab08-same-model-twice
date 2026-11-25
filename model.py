"""Neural network for predicting hourly bike rentals."""

import torch
from torch import nn


class BikeRentalNetwork(nn.Module):
    """Feed-forward neural network with configurable hidden layers."""

    def __init__(self, input_size=8, hidden_sizes=(64, 32)):
        super().__init__()

        layers = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, features):
        """Return one prediction per observation."""
        return self.network(features).squeeze(-1)


if __name__ == "__main__":
    model = BikeRentalNetwork()

    sample = torch.randn(64, 8)
    predictions = model(sample)

    print(model)
    print("Input shape:", sample.shape)
    print("Output shape:", predictions.shape)
