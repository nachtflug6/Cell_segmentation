# https://iq.opengenus.org/inception-resnet-v1/

import numpy as np

import torch as th
import torch.nn as nn

from .base_block import BaseBlock


class InceptionResNetModuleA(BaseBlock):
    def __init__(self, in_ch, out_ch, padding_mode='zeros'):
        super(BaseBlock, self).__init__()

        inter_ch = in_ch // 4

        self.relu = nn.ReLU()

        self.conv_path1 = nn.Conv2d(in_ch, inter_ch, kernel_size=1, padding=0, stride=1)

        self.conv_path2 = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=3, padding=1, stride=1, padding_mode=padding_mode))

        self.conv_path3 = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=3, padding=1, stride=1, padding_mode=padding_mode),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=3, padding=1, stride=1, padding_mode=padding_mode))

        self.conv_out = nn.Conv2d(3 * inter_ch, out_ch, kernel_size=1, padding=0, stride=1)

    def forward(self, input_tensor):
        input_tensor = self.relu(input_tensor)

        output_1_1 = self.conv_path1(input_tensor)
        output_1_2 = self.conv_path2(input_tensor)
        output_1_3 = self.conv_path3(input_tensor)
        output_1 = th.cat((output_1_1, output_1_2, output_1_3), dim=1)

        output_2 = self.conv_out(output_1)

        output = self.relu(output_2 + input_tensor)

        return output


class InceptionResNetModuleB(BaseBlock):
    def __init__(self, in_ch, out_ch, padding_mode='zeros'):
        super(BaseBlock, self).__init__()

        inter_ch = in_ch // 4

        self.relu = nn.ReLU()

        self.conv_path1 = nn.Conv2d(in_ch, inter_ch, kernel_size=1, padding=0, stride=1)

        self.conv_path2 = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=(1, 7), padding=(0, 3), stride=1, padding_mode=padding_mode),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=(7, 1), padding=(3, 0), stride=1, padding_mode=padding_mode))

        self.conv_out = nn.Conv2d(2 * inter_ch, out_ch, kernel_size=1, padding=0, stride=1)

    def forward(self, input_tensor):
        input_tensor = self.relu(input_tensor)

        output_1_1 = self.conv_path1(input_tensor)
        output_1_2 = self.conv_path2(input_tensor)
        output_1 = th.cat((output_1_1, output_1_2), dim=1)

        output_2 = self.conv_out(output_1)

        return self.relu(output_2 + input_tensor)


class InceptionResNetModuleC(BaseBlock):
    def __init__(self, in_ch, out_ch, padding_mode='zeros'):
        super(BaseBlock, self).__init__()

        inter_ch = in_ch // 4

        self.relu = nn.ReLU()

        self.conv_path1 = nn.Conv2d(in_ch, inter_ch, kernel_size=1, padding=0, stride=1)

        self.conv_path2 = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=(1, 3), padding=(0, 1), stride=1, padding_mode=padding_mode),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=(3, 1), padding=(1, 0), stride=1, padding_mode=padding_mode))

        self.conv_out = nn.Conv2d(2 * inter_ch, in_ch, kernel_size=1, padding=0, stride=1)

    def forward(self, input_tensor):
        input_tensor = self.relu(input_tensor)

        output_1_1 = self.conv_path1(input_tensor)
        output_1_2 = self.conv_path2(input_tensor)
        output_1 = th.cat((output_1_1, output_1_2), dim=1)

        output_2 = self.conv_out(output_1)

        return self.relu(output_2 + input_tensor)


class InceptionModuleA(BaseBlock):
    def __init__(self, in_ch, padding_mode='zeros'):
        super(BaseBlock, self).__init__()

        inter_ch = in_ch // 2

        self.relu = nn.ReLU()

        self.conv_path1 = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=3, padding=1, stride=1, padding_mode=padding_mode)
        )

        self.conv_path2 = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, padding=0, stride=1),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=(7, 1), padding=(3, 0), stride=1, padding_mode=padding_mode),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=(1, 7), padding=(0, 3), stride=1, padding_mode=padding_mode),
            nn.Conv2d(inter_ch, inter_ch, kernel_size=3, padding=1, stride=1, padding_mode=padding_mode)
        )

        self.conv_out = nn.Conv2d(2 * inter_ch, in_ch, kernel_size=1, padding=0, stride=1)

    def forward(self, input_tensor):
        input_tensor = self.relu(input_tensor)

        output_1_1 = self.conv_path1(input_tensor)
        output_1_2 = self.conv_path2(input_tensor)
        output = th.cat((output_1_1, output_1_2), dim=1)

        return self.relu(output)

# class InceptionResNetBlockAssembly(BaseBlock):
#     def __init__(self, io_ch, constellation, padding_mode='zeros'):
#         super(BaseBlock, self).__init__()
#
#         output_list = []
#
#         for char in constellation:
#
#             current_object = None
#
#             if char == 'a':
#                 current_object = InceptionResNetModuleA(io_ch, padding_mode)
#             if char == 'b':
#                 current_object = InceptionResNetModuleB(io_ch, padding_mode)
#             if char == 'c':
#                 current_object = InceptionResNetModuleC(io_ch, padding_mode)
#
#             output_list.append(current_object)
#
#         self.pipeline = nn.Sequential(*tuple(output_list))
#
#     def forward(self, input_tensor):
#         output = self.pipeline(input_tensor)
#         return output
