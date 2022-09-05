# https://towardsdev.com/original-u-net-in-pytorch-ebe7bb705cc7
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
# https://tuatini.me/practical-image-segmentation-with-unet/
# https://discuss.pytorch.org/t/3d-unet-patch-based-segmentation-output-artifacts/60980/2

import torch as th
import torch.nn as nn
from .blocks.unet_blocks import *


class UNet(nn.Module):
    def __init__(self, params):
        super(UNet, self).__init__()
        self.batch_1 = th.nn.BatchNorm2d(64)
        self.batch_2 = th.nn.BatchNorm2d(128)
        self.batch_3 = th.nn.BatchNorm2d(256)
        self.batch_4 = th.nn.BatchNorm2d(512)
        self.batch_5 = th.nn.BatchNorm2d(1024)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn1_up = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn2_up = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn3_up = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn4_up = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
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
        output_1 = self.batch_1(output_1)

        output_2 = self.cnn2(self.max_pool_2(output_1))
        output_2 = self.batch_2(output_2)

        output_3 = self.cnn3(self.max_pool_2(output_2))
        output_3 = self.batch_3(output_3)

        output_4 = self.cnn4(self.max_pool_2(output_3))
        output_4 = self.batch_4(output_4)

        output_5 = self.cnn5(self.max_pool_2(output_4))
        output_5 = self.batch_5(output_5)

        output_1_up = self.cnn1_up(th.cat((self.upconv_1(output_5), output_4), dim=1))
        output_1_up = self.batch_4(output_1_up)

        output_2_up = self.cnn2_up(th.cat((self.upconv_2(output_1_up), output_3), dim=1))
        output_2_up = self.batch_3(output_2_up)

        output_3_up = self.cnn3_up(th.cat((self.upconv_3(output_2_up), output_2), dim=1))
        output_3_up = self.batch_2(output_3_up)

        output = self.cnn4_up(th.cat((self.upconv_4(output_3_up), output_1), dim=1))

        output = self.softmax(output)

        return output


class UNetSmall(nn.Module):
    def __init__(self, params):
        super(UNetSmall, self).__init__()
        self.batch_1 = th.nn.BatchNorm2d(64)
        self.batch_2 = th.nn.BatchNorm2d(128)
        self.batch_3 = th.nn.BatchNorm2d(256)
        self.batch_4 = th.nn.BatchNorm2d(512)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn2_up = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn3_up = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn4_up = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
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
        output_1 = self.batch_1(output_1)

        output_2 = self.cnn2(self.max_pool_2(output_1))
        output_2 = self.batch_2(output_2)

        output_3 = self.cnn3(self.max_pool_2(output_2))
        output_3 = self.batch_3(output_3)

        output_4 = self.cnn4(self.max_pool_2(output_3))
        output_4 = self.batch_4(output_4)

        output_2_up = self.cnn2_up(th.cat((self.upconv_2(output_4), output_3), dim=1))
        output_2_up = self.batch_3(output_2_up)

        output_3_up = self.cnn3_up(th.cat((self.upconv_3(output_2_up), output_2), dim=1))
        output_3_up = self.batch_2(output_3_up)

        output = self.cnn4_up(th.cat((self.upconv_4(output_3_up), output_1), dim=1))

        output = self.softmax(output)

        return output


class UNetLarge(nn.Module):
    def __init__(self, params):
        super(UNetLarge, self).__init__()
        self.batch_1 = th.nn.BatchNorm2d(64)
        self.batch_2 = th.nn.BatchNorm2d(128)
        self.batch_3 = th.nn.BatchNorm2d(256)
        self.batch_4 = th.nn.BatchNorm2d(512)
        self.batch_5 = th.nn.BatchNorm2d(1024)
        self.batch_6 = th.nn.BatchNorm2d(2048)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn6 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn0_up = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn1_up = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn2_up = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn3_up = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU())

        self.cnn4_up = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode=params['padding_mode']),
            nn.ReLU(),
            nn.Conv2d(64, params['out_classes'], kernel_size=1, stride=1, padding=0))

        self.max_pool_2 = nn.MaxPool2d(2)

        self.upconv_0 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.upconv_1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.upconv_2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.upconv_3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv_4 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor):
        output_1 = self.cnn1(input_tensor)
        output_1 = self.batch_1(output_1)

        output_2 = self.cnn2(self.max_pool_2(output_1))
        output_2 = self.batch_2(output_2)

        output_3 = self.cnn3(self.max_pool_2(output_2))
        output_3 = self.batch_3(output_3)

        output_4 = self.cnn4(self.max_pool_2(output_3))
        output_4 = self.batch_4(output_4)

        output_5 = self.cnn5(self.max_pool_2(output_4))
        output_5 = self.batch_5(output_5)

        output_6 = self.cnn6(self.max_pool_2(output_5))
        output_6 = self.batch_6(output_6)

        output_0_up = self.cnn0_up(th.cat((self.upconv_0(output_6), output_5), dim=1))
        output_0_up = self.batch_5(output_0_up)

        output_1_up = self.cnn1_up(th.cat((self.upconv_1(output_0_up), output_4), dim=1))
        output_1_up = self.batch_4(output_1_up)

        output_2_up = self.cnn2_up(th.cat((self.upconv_2(output_1_up), output_3), dim=1))
        output_2_up = self.batch_3(output_2_up)

        output_3_up = self.cnn3_up(th.cat((self.upconv_3(output_2_up), output_2), dim=1))
        output_3_up = self.batch_2(output_3_up)

        output = self.cnn4_up(th.cat((self.upconv_4(output_3_up), output_1), dim=1))

        output = self.softmax(output)

        return output