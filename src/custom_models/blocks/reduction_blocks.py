import numpy as np

import torch as th
import torch.nn as nn

from .base_block import BaseBlock


class IncResReductionBy2ModuleA(BaseBlock):
    def __init__(self, in_ch, out_ch, padding_mode='zeros'):
        super(BaseBlock, self).__init__()

        self.pool_path1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.pool_path2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode)

        self.pool_path3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=1, padding_mode=padding_mode),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode))

        self.pool_path4 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))

        self.conv_out = nn.Conv2d(4 * in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor):
        output_1_1 = self.pool_path1(input_tensor)
        output_1_2 = self.pool_path2(input_tensor)
        output_1_3 = self.pool_path3(input_tensor)
        output_1_4 = self.pool_path4(input_tensor)
        output_1 = th.cat((output_1_1, output_1_2, output_1_3, output_1_4), dim=1)

        output = self.conv_out(output_1)

        return output


class SimpleReductionBy2(BaseBlock):
    def __init__(self, in_ch, padding_mode='zeros'):
        super(BaseBlock, self).__init__()

        self.pool_path1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_path2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=2, padding_mode=padding_mode)

    def forward(self, input_tensor):
        output_1_1 = self.pool_path1(input_tensor)
        output_1_2 = self.pool_path2(input_tensor)

        return th.cat((output_1_1, output_1_2), dim=1)

