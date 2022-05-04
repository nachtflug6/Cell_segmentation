import numpy as np

import torch as th
import torch.nn as nn

from models.blocks.constant_blocks import *
from models.blocks.reduction_blocks import *
from models.blocks.unet_blocks import *
from models.base_model import BaseModel


class UNetResInc(BaseModel):
    def __init__(self, params):
        super(BaseModel, self).__init__()
        self.depth = params['depth']
        self.out_classes = params['out_classes']


        self.input_layer = nn.Sequential(
            nn.Conv2d(1, params['start_layers'], kernel_size=params['input_conv_kernel_size'],
                      stride=1, padding=params['input_conv_kernel_size'] // 2,
                      padding_mode=params['padding_mode']),
            nn.ReLU()
        )

        start_layers = params['start_layers']
        self.start_layers = start_layers

        down_blocks = []
        up_blocks = []

        for i in range(self.depth):
            down_block = DownBlock(
                nn.Sequential(InceptionResNetModuleA(start_layers, padding_mode=params['padding_mode']),
                              nn.Dropout(params['dropout_p']),
                              nn.BatchNorm2d(start_layers)
                              ),
                nn.Sequential(ReductionModuleA(start_layers, padding_mode=params['padding_mode'])))

            up_block = UpBlock(
                nn.Sequential(nn.ConvTranspose2d(start_layers * 2, start_layers, 2, stride=2)),
                nn.Sequential(nn.Conv2d(2 * start_layers, start_layers, kernel_size=1, stride=1, padding=0),
                              InceptionResNetModuleA(start_layers, padding_mode=params['padding_mode']),
                              nn.Dropout(params['dropout_p']),
                              nn.BatchNorm2d(start_layers)
                              ))

            start_layers *= 2

            down_blocks.append(down_block)
            up_blocks.append(up_block)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)

        self.buttom_layer = nn.Sequential(InceptionResNetModuleA(start_layers, padding_mode=params['padding_mode']),
                                          nn.Dropout(params['dropout_p']),
                                          nn.BatchNorm2d(start_layers)
                                          )

        self.output_layer1 = nn.Sequential(
            InceptionResNetModuleA(params['start_layers'], padding_mode=params['padding_mode']),
            InceptionResNetModuleB(params['start_layers'], padding_mode=params['padding_mode']),
            InceptionResNetModuleC(params['start_layers'], padding_mode=params['padding_mode']),
            nn.Dropout(params['dropout_p']),
            nn.BatchNorm2d(params['start_layers']),
            #nn.Conv2d(params['start_layers'], params['out_classes'], kernel_size=1, stride=1, padding=0)
        )

        self.output_layer2 = nn.Sequential(
            nn.Conv2d(params['start_layers'], params['out_classes'], kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1))

    def forward(self, input_tensor):

        current_output = self.input_layer(input_tensor)
        #input_tensor = current_output
        concat_outputs = []

        for down_block in self.down_blocks:
            concat_output, current_output = down_block.forward(current_output)
            concat_outputs.append(concat_output)

        current_output = self.buttom_layer.forward(current_output)

        for i, up_block in enumerate(reversed(self.up_blocks)):

            idx = self.depth - 1 - i
            current_output = up_block.forward(current_output, concat_outputs[idx])

        output = self.output_layer1(current_output)

        output = output + input_tensor.expand(-1, self.start_layers, -1, -1)

        output = self.output_layer2(output)

        return output
