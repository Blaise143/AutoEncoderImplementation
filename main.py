import torch
import torch.nn as nn
import numpy as np
from typing import Type

from torch import Tensor


class StackedAutoencoder(nn.Module):
    """
    This is a class for an autoencoder Notes about autoencoders:
        - They learn to copy inputs to outputs
        - There is a need to add some noise in order to make sure
          the neuralnet is not simply copying but rather learning a representation of the data
        - The autoencoder takes in an input, converts it to a latent representation and then spits
          out an output that hopefully is similar to the input
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()

        encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=100),
            nn.SELU(),
            nn.Linear(in_features=100, out_features=30),
            nn.SELU()
        )

        decoder = nn.Sequential(
            nn.Linear(30, 100),
            nn.SELU(),
            nn.Linear(in_features=100, out_features=in_features),
        )

        self.blocks = nn.Sequential(encoder, decoder)

    def forward(self, x) -> Tensor:
        """
        perform the forward pass
        :param x: then input image
        :return: a tensor image
        """
        x_shape = x.shape
        x = self.blocks(x)
        x = torch.reshape(x, x_shape)
        return x
