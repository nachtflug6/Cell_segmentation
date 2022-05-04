import numpy as np

import torch as th
import torch.nn as nn

from .base_block import BaseBlock


class DownBlock(nn.Module):
    def __init__(self, maintain_layers, reduction_layers):
        super(DownBlock, self).__init__()
        self.maintain_layers = maintain_layers
        self.reduction_layers = reduction_layers

    def forward(self, input_tensor):
        maintained_output = self.maintain_layers(input_tensor)
        reduced_output = self.reduction_layers(maintained_output)
        return maintained_output, reduced_output


class UpBlock(nn.Module):
    def __init__(self, expand_layers, maintain_layers):
        super(UpBlock, self).__init__()
        self.expand_layers = expand_layers
        self.maintain_layers = maintain_layers

    def forward(self, reduced_input_tensor, maintained_input_tensor):
        expanded_output = self.expand_layers(reduced_input_tensor)
        input_tensor = th.cat((maintained_input_tensor, expanded_output), dim=1)
        output = self.maintain_layers(input_tensor)
        return output
