# https://towardsdev.com/original-u-net-in-pytorch-ebe7bb705cc7
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
# https://tuatini.me/practical-image-segmentation-with-unet/
# https://discuss.pytorch.org/t/3d-unet-patch-based-segmentation-output-artifacts/60980/2
import numpy as np

import torch as th
import torch.nn as nn

from .base_model import BaseModel


class UNet(BaseModel):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.batch_1 = th.nn.BatchNorm2d(64)
        self.batch_2 = th.nn.BatchNorm2d(128)
        self.batch_3 = th.nn.BatchNorm2d(256)
        self.batch_4 = th.nn.BatchNorm2d(512)
        self.batch_5 = th.nn.BatchNorm2d(1024)
        self.batch_out = th.nn.BatchNorm2d(2)

        self.dropout = nn.Dropout(p=0.2)

        self.cnn1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                  nn.ReLU())

        self.cnn2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                  nn.ReLU())

        self.cnn3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                  nn.ReLU(),
                                  nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                  nn.ReLU())

        self.cnn4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                  nn.ReLU(),
                                  nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                  nn.ReLU())

        self.cnn5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                  nn.ReLU(),
                                  nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                  nn.ReLU())

        self.cnn1_up = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                     nn.ReLU(),
                                     nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                     nn.ReLU())

        self.cnn2_up = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                     nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                     nn.ReLU())

        self.cnn3_up = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                     nn.ReLU())

        self.cnn4_up = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
                                     nn.ReLU(),
                                     nn.Conv2d(64, params['out_classes'], kernel_size=1, stride=1, padding=0))

        self.max_pool_2 = nn.MaxPool2d(2)

        self.upconv_1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.upconv_2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv_3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv_4 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor):

        output_1 = self.cnn1(input_tensor)
        output_1 = self.dropout(output_1)
        output_1 = self.batch_1(output_1)

        output_2 = self.cnn2(self.max_pool_2(output_1))
        output_2 = self.dropout(output_2)
        output_2 = self.batch_2(output_2)

        output_3 = self.cnn3(self.max_pool_2(output_2))
        output_3 = self.dropout(output_3)
        output_3 = self.batch_3(output_3)

        output_4 = self.cnn4(self.max_pool_2(output_3))
        output_4 = self.dropout(output_4)
        output_4 = self.batch_4(output_4)

        output_5 = self.cnn5(self.max_pool_2((output_4)))
        output_5 = self.dropout(output_5)
        output_5 = self.batch_5(output_5)

        output_1_up = self.cnn1_up(th.cat((self.upconv_1(output_5), output_4), dim=1))
        output_1_up = self.dropout(output_1_up)
        output_1_up = self.batch_4(output_1_up)

        output_2_up = self.cnn2_up(th.cat((self.upconv_2(output_1_up), output_3), dim=1))
        output_2_up = self.dropout(output_2_up)
        output_2_up = self.batch_3(output_2_up)

        output_3_up = self.cnn3_up(th.cat((self.upconv_3(output_2_up), output_2), dim=1))
        output_3_up = self.dropout(output_3_up)
        output_3_up = self.batch_2(output_3_up)

        output = self.cnn4_up(th.cat((self.upconv_4(output_3_up), output_1), dim=1))
        # output = self.dropout(output)
        # output = self.batch_out(output)

        output = self.softmax(output)

        return output




