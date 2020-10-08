from layer import *

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision import models

import torch
import torch.nn as nn

from padding import *
import torch.nn.functional as F

resnet18_base_model = models.resnet18(pretrained=False)


def convrelu(in_channels, out_channels, kernel, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels,
                  out_channels,
                  kernel,
                  stride=stride,
                  padding=padding), nn.ReLU(inplace=True))


class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)

        self.base_layers = list(resnet18_base_model.children())

        self.layer0 = nn.Sequential(
            *self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(
            *self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out


class CNP(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(CNP, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True
        """
        Encoder part
        """

        self.enc1_1 = CNR2d(1 * self.nch_in,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc1_2 = CNR2d(1 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc2_2 = CNR2d(2 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc3_2 = CNR2d(4 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(4 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc4_2 = CNR2d(8 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(8 * self.nch_ker,
                            16 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        """
        Skip part
        """

        self.skip1 = CNR2d(1 * self.nch_ker,
                           1 * self.nch_ker,
                           kernel_size=3,
                           stride=1,
                           norm=self.norm,
                           relu=0.0)
        self.skip2 = CNR2d(2 * self.nch_ker,
                           2 * self.nch_ker,
                           kernel_size=3,
                           stride=1,
                           norm=self.norm,
                           relu=0.0)
        self.skip3 = CNR2d(4 * self.nch_ker,
                           4 * self.nch_ker,
                           kernel_size=3,
                           stride=1,
                           norm=self.norm,
                           relu=0.0)
        self.skip4 = CNR2d(8 * self.nch_ker,
                           8 * self.nch_ker,
                           kernel_size=3,
                           stride=1,
                           norm=self.norm,
                           relu=0.0)

        # self.skip1 = CBAM(1 * self.nch_ker, 1 * self.nch_ker // 4)
        # self.skip2 = CBAM(2 * self.nch_ker, 2 * self.nch_ker // 4)
        # self.skip3 = CBAM(4 * self.nch_ker, 4 * self.nch_ker // 4)
        # self.skip4 = CBAM(8 * self.nch_ker, 8 * self.nch_ker // 4)
        """
        Decoder part
        """
        self.dec5_1 = CNR2d(16 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool4 = UnPooling2d(pool=2, type='nearest')
        # self.unpool4 = UnPooling2d(pool=2, type='bilinear')
        self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='conv')

        self.dec4_2 = CNR2d(2 * 8 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec4_1 = CNR2d(8 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool3 = UnPooling2d(pool=2, type='nearest')
        # self.unpool3 = UnPooling2d(pool=2, type='bilinear')
        self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='conv')

        self.dec3_2 = CNR2d(2 * 4 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec3_1 = CNR2d(4 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool2 = UnPooling2d(pool=2, type='nearest')
        # self.unpool2 = UnPooling2d(pool=2, type='bilinear')
        self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='conv')

        self.dec2_2 = CNR2d(2 * 2 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec2_1 = CNR2d(2 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool1 = UnPooling2d(pool=2, type='nearest')
        # self.unpool1 = UnPooling2d(pool=2, type='bilinear')
        self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='conv')

        self.dec1_2 = CNR2d(2 * 1 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec1_1 = CNR2d(1 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.fc = Conv2d(1 * self.nch_ker,
                         1 * self.nch_out,
                         kernel_size=1,
                         padding=0)

    def forward(self, x):
        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)
        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        skip4 = self.skip4(enc4)
        cat4 = torch.cat([skip4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))
        # cat4 = torch.cat([enc4, unpool4], dim=1)
        # dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        skip3 = self.skip3(enc3)
        cat3 = torch.cat([skip3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))
        # cat3 = torch.cat([enc3, unpool3], dim=1)
        # dec3 = self.dec3_1(self.dec3_2(cat3))

        unpool2 = self.unpool2(dec3)
        skip2 = self.skip2(enc2)
        cat2 = torch.cat([skip2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))
        # cat2 = torch.cat([enc2, unpool2], dim=1)
        # dec2 = self.dec2_1(self.dec2_2(cat2))

        unpool1 = self.unpool1(dec2)
        skip1 = self.skip1(enc1)
        cat1 = torch.cat([skip1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))
        # cat1 = torch.cat([enc1, unpool1], dim=1)
        # dec1 = self.dec1_1(self.dec1_2(cat1))

        x = self.fc(dec1)

        # x = torch.sigmoid(x)
        # x = dec1

        return x


class UNet(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(UNet, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True
        """
        Encoder part
        """

        self.enc1_1 = CNR2d(1 * self.nch_in,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc1_2 = CNR2d(1 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc2_2 = CNR2d(2 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc3_2 = CNR2d(4 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(4 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc4_2 = CNR2d(8 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(8 * self.nch_ker,
                            2 * 8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        """
        Decoder part
        """
        self.dec5_1 = CNR2d(16 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool4 = UnPooling2d(pool=2, type='nearest')
        # self.unpool4 = UnPooling2d(pool=2, type='bilinear')
        self.unpool4 = UnPooling2d(nch=8 * self.nch_ker, pool=2, type='conv')

        self.dec4_2 = CNR2d(2 * 8 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec4_1 = CNR2d(8 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool3 = UnPooling2d(pool=2, type='nearest')
        # self.unpool3 = UnPooling2d(pool=2, type='bilinear')
        self.unpool3 = UnPooling2d(nch=4 * self.nch_ker, pool=2, type='conv')

        self.dec3_2 = CNR2d(2 * 4 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec3_1 = CNR2d(4 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool2 = UnPooling2d(pool=2, type='nearest')
        # self.unpool2 = UnPooling2d(pool=2, type='bilinear')
        self.unpool2 = UnPooling2d(nch=2 * self.nch_ker, pool=2, type='conv')

        self.dec2_2 = CNR2d(2 * 2 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec2_1 = CNR2d(2 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool1 = UnPooling2d(pool=2, type='nearest')
        # self.unpool1 = UnPooling2d(pool=2, type='bilinear')
        self.unpool1 = UnPooling2d(nch=1 * self.nch_ker, pool=2, type='conv')

        self.dec1_2 = CNR2d(2 * 1 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec1_1 = CNR2d(1 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.fc = Conv2d(1 * self.nch_ker,
                         1 * self.nch_out,
                         kernel_size=1,
                         padding=0)

    def forward(self, x):
        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)
        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(cat4))

        unpool3 = self.unpool3(dec4)
        cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(cat3))

        unpool2 = self.unpool2(dec3)
        cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(cat2))

        unpool1 = self.unpool1(dec2)
        cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(cat1))

        x = self.fc(dec1)
        # x = torch.sigmoid(x)

        return x


class Hourglass(nn.Module):
    def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm'):
        super(Hourglass, self).__init__()

        self.nch_in = nch_in
        self.nch_out = nch_out
        self.nch_ker = nch_ker
        self.norm = norm

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True
        """
        Encoder part
        """

        self.enc1_1 = CNR2d(1 * self.nch_in,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc1_2 = CNR2d(1 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool1 = Pooling2d(pool=2, type='avg')

        self.enc2_1 = CNR2d(1 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc2_2 = CNR2d(2 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool2 = Pooling2d(pool=2, type='avg')

        self.enc3_1 = CNR2d(2 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc3_2 = CNR2d(4 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool3 = Pooling2d(pool=2, type='avg')

        self.enc4_1 = CNR2d(4 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.enc4_2 = CNR2d(8 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.pool4 = Pooling2d(pool=2, type='avg')

        self.enc5_1 = CNR2d(8 * self.nch_ker,
                            2 * 8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        """
        Decoder part
        """

        self.dec5_1 = CNR2d(2 * 8 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool4 = UnPooling2d(pool=2, type='nearest')
        self.unpool4 = UnPooling2d(pool=2, type='bilinear')

        self.dec4_2 = CNR2d(1 * 8 * self.nch_ker,
                            8 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec4_1 = CNR2d(8 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool3 = UnPooling2d(pool=2, type='nearest')
        self.unpool3 = UnPooling2d(pool=2, type='bilinear')

        self.dec3_2 = CNR2d(1 * 4 * self.nch_ker,
                            4 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec3_1 = CNR2d(4 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool2 = UnPooling2d(pool=2, type='nearest')
        self.unpool2 = UnPooling2d(pool=2, type='bilinear')

        self.dec2_2 = CNR2d(1 * 2 * self.nch_ker,
                            2 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec2_1 = CNR2d(2 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        # self.unpool1 = UnPooling2d(pool=2, type='nearest')
        self.unpool1 = UnPooling2d(pool=2, type='bilinear')

        self.dec1_2 = CNR2d(1 * 1 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)
        self.dec1_1 = CNR2d(1 * self.nch_ker,
                            1 * self.nch_ker,
                            kernel_size=3,
                            stride=1,
                            norm=self.norm,
                            relu=0.0)

        self.fc = Conv2d(1 * self.nch_ker,
                         1 * self.nch_out,
                         kernel_size=1,
                         padding=0)

    def forward(self, x):
        """
        Encoder part
        """

        enc1 = self.enc1_2(self.enc1_1(x))
        pool1 = self.pool1(enc1)

        enc2 = self.enc2_2(self.enc2_1(pool1))
        pool2 = self.pool2(enc2)

        enc3 = self.enc3_2(self.enc3_1(pool2))
        pool3 = self.pool3(enc3)

        enc4 = self.enc4_2(self.enc4_1(pool3))
        pool4 = self.pool4(enc4)

        enc5 = self.enc5_1(pool4)
        """
        Encoder part
        """
        dec5 = self.dec5_1(enc5)

        unpool4 = self.unpool4(dec5)
        # cat4 = torch.cat([enc4, unpool4], dim=1)
        dec4 = self.dec4_1(self.dec4_2(unpool4))

        unpool3 = self.unpool3(dec4)
        # cat3 = torch.cat([enc3, unpool3], dim=1)
        dec3 = self.dec3_1(self.dec3_2(unpool3))

        unpool2 = self.unpool2(dec3)
        # cat2 = torch.cat([enc2, unpool2], dim=1)
        dec2 = self.dec2_1(self.dec2_2(unpool2))

        unpool1 = self.unpool1(dec2)
        # cat1 = torch.cat([enc1, unpool1], dim=1)
        dec1 = self.dec1_1(self.dec1_2(unpool1))

        x = self.fc(dec1)
        # x = torch.sigmoid(x)

        return x


# class ResNet(nn.Module):
#     def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=6):
#         super(ResNet, self).__init__()
#
#         self.nch_in = nch_in
#         self.nch_out = nch_out
#         self.nch_ker = nch_ker
#         self.norm = norm
#         self.nblk = nblk
#
#         if norm == 'bnorm':
#             self.bias = False
#         else:
#             self.bias = True
#
#         self.enc1 = CNR2d(self.nch_in,      1 * self.nch_ker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
#
#         self.enc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
#
#         self.enc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
#
#         if self.nblk:
#             res = []
#
#             for i in range(self.nblk):
#                 res += [ResBlock(4 * self.nch_ker, 4 * self.nch_ker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0, padding_mode='reflection')]
#
#             self.res = nn.Sequential(*res)
#
#         self.dec3 = DECNR2d(4 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
#
#         self.dec2 = DECNR2d(2 * self.nch_ker, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm, relu=0.0)
#
#         self.dec1 = CNR2d(1 * self.nch_ker, self.nch_out, kernel_size=7, stride=1, padding=3, norm=[], relu=[], bias=False)
#
#     def forward(self, x):
#         x = self.enc1(x)
#         x = self.enc2(x)
#         x = self.enc3(x)
#
#         if self.nblk:
#             x = self.res(x)
#
#         x = self.dec3(x)
#         x = self.dec2(x)
#         x = self.dec1(x)
#
#         x = torch.tanh(x)
#
#         return x

# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
# https://arxiv.org/abs/1609.04802
#  class ResNet(nn.Module):
#     def __init__(self, nch_in, nch_out, nch_ker=64, norm='bnorm', nblk=16):
#         super(ResNet, self).__init__()

#         self.nch_in = nch_in
#         self.nch_out = nch_out
#         self.nch_ker = nch_ker
#         self.norm = norm
#         self.nblk = nblk

#         if norm == 'bnorm':
#             self.bias = False
#         else:
#             self.bias = True

#         self.enc1 = CNR2d(self.nch_in,
#                           self.nch_ker,
#                           kernel_size=3,
#                           stride=1,
#                           padding=1,
#                           norm=[],
#                           relu=0.0)

#         res = []
#         for i in range(self.nblk):
#             res += [
#                 ResBlock(self.nch_ker,
#                          self.nch_ker,
#                          kernel_size=3,
#                          stride=1,
#                          padding=1,
#                          norm=self.norm,
#                          relu=0.0,
#                          padding_mode='reflection')
#             ]
#         self.res = nn.Sequential(*res)

#         self.dec1 = CNR2d(self.nch_ker,
#                           self.nch_ker,
#                           kernel_size=3,
#                           stride=1,
#                           padding=1,
#                           norm=norm,
#                           relu=[])

#         self.conv1 = Conv2d(self.nch_ker,
#                             self.nch_out,
#                             kernel_size=3,
#                             stride=1,
#                             padding=1)

#     def forward(self, x):
#         x = self.enc1(x)
#         x0 = x

#         x = self.res(x)

#         x = self.dec1(x)
#         x = x + x0

#         x = self.conv1(x)

#         return x

# class Discriminator(nn.Module):
#     def __init__(self, nch_in, nch_ker=64, norm='bnorm'):
#         super(Discriminator, self).__init__()

#         self.nch_in = nch_in
#         self.nch_ker = nch_ker
#         self.norm = norm

#         if norm == 'bnorm':
#             self.bias = False
#         else:
#             self.bias = True

#         # dsc1 : 256 x 256 x 3 -> 128 x 128 x 64
#         # dsc2 : 128 x 128 x 64 -> 64 x 64 x 128
#         # dsc3 : 64 x 64 x 128 -> 32 x 32 x 256
#         # dsc4 : 32 x 32 x 256 -> 16 x 16 x 512
#         # dsc5 : 16 x 16 x 512 -> 16 x 16 x 1

#         self.dsc1 = CNR2d(1 * self.nch_in,
#                           1 * self.nch_ker,
#                           kernel_size=4,
#                           stride=2,
#                           padding=1,
#                           norm=self.norm,
#                           relu=0.2)
#         self.dsc2 = CNR2d(1 * self.nch_ker,
#                           2 * self.nch_ker,
#                           kernel_size=4,
#                           stride=2,
#                           padding=1,
#                           norm=self.norm,
#                           relu=0.2)
#         self.dsc3 = CNR2d(2 * self.nch_ker,
#                           4 * self.nch_ker,
#                           kernel_size=4,
#                           stride=2,
#                           padding=1,
#                           norm=self.norm,
#                           relu=0.2)
#         self.dsc4 = CNR2d(4 * self.nch_ker,
#                           8 * self.nch_ker,
#                           kernel_size=4,
#                           stride=2,
#                           padding=1,
#                           norm=self.norm,
#                           relu=0.2)
#         self.dsc5 = CNR2d(8 * self.nch_ker,
#                           1,
#                           kernel_size=4,
#                           stride=1,
#                           padding=1,
#                           norm=[],
#                           relu=[],
#                           bias=False)

#         # self.dsc1 = CNR2d(1 * self.nch_in,  1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
#         # self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
#         # self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=[], relu=0.2)
#         # self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=1, padding=1, norm=[], relu=0.2)
#         # self.dsc5 = CNR2d(8 * self.nch_ker, 1,                kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False)

#     def forward(self, x):

#         x = self.dsc1(x)
#         x = self.dsc2(x)
#         x = self.dsc3(x)
#         x = self.dsc4(x)
#         x = self.dsc5(x)

#         # x = torch.sigmoid(x)

#         return x


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain)
    return net



class Conv2dBatchNorm(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
        super(Conv2dBatchNorm,self).__init__()
        self.conv2dn = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self,x):
        result = self.conv2dn(x)
        return result

class ResBlockA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation):
        super(ResBlockA, self).__init__()

        same = get_padding(kernel_size,stride,dilation)
        # same = 1

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      dilation=dilation,
                      padding=same),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      dilation=dilation,
                      padding=same))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = same

    def forward(self, x):
        x1 = self.conv_block(x)
        return x1


class ResBlockAD4(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, num):
        super(ResBlockAD4, self).__init__()

        # print(f'in_ch {in_channels} out_ch: {out_channels} dilations: {dilations[0]} Block n th: {num}')

        self.ResBlock1 = ResBlockA(in_channels,
                                   out_channels,
                                   3,
                                   1,
                                   dilation=dilations[0])
        self.ResBlock2 = ResBlockA(in_channels,
                                   out_channels,
                                   3,
                                   1,
                                   dilation=dilations[1])
        self.ResBlock3 = ResBlockA(in_channels,
                                   out_channels,
                                   3,
                                   1,
                                   dilation=dilations[2])
        self.ResBlock4 = ResBlockA(in_channels,
                                   out_channels,
                                   3,
                                   1,
                                   dilation=dilations[3])
        self.num = num

    def forward(self, x):
        x1 = self.ResBlock1(x)
        x2 = self.ResBlock2(x)
        x3 = self.ResBlock3(x)
        x4 = self.ResBlock4(x)

        x12 = x1.add(x2)
        x34 = x3.add(x4)
        xAll = x12.add(x34)

        return xAll


class ResBlockAD3(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(ResBlockAD3, self).__init__()
        print()
        self.ResBlock1 = ResBlockA(in_channels, out_channels, 3, 1,
                                   dilations[0])
        self.ResBlock2 = ResBlockA(in_channels, out_channels, 3, 1,
                                   dilations[1])
        self.ResBlock3 = ResBlockA(in_channels, out_channels, 3, 1,
                                   dilations[2])

    def forward(self, x):
        x1 = self.ResBlock1(x)
        x2 = self.ResBlock2(x)
        x3 = self.ResBlock2(x)
        x12 = torch.add(x1, x2)
        xAll = torch.add(x12,x3)
        return xAll


class ResBlockAD1(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(ResBlockAD1, self).__init__()
        self.ResBlock1 = ResBlockA(in_channels, out_channels, 3, 1,
                                   dilations[0])

    def forward(self, x):
        x1 = self.ResBlock1(x)
        return x1


class Combine(nn.Module):
    def __init__(self,input_size):
        super(Combine, self).__init__()
        self.input_size = input_size
        # print(f'input size: {input_size}')
        self.convn = nn.Sequential(
            nn.Conv2d(input_size * 2,input_size, 1),
            nn.BatchNorm2d(input_size)
        )

        self.relu = nn.ReLU(True)

    def forward(self, input1, input2):
        input1_relu = self.relu(input1.cuda())
        # print(f'input1 relu size : {input1_relu.size()} input2 size : {input2.size()}')
        input_concat = torch.cat([input1_relu, input2],dim=1)
        # print(f'input_concat size : {input_concat.size()}')
        output_conv = self.convn(input_concat.cuda())
        # return output_conv
        return output_conv



def _PSP1x1Conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        # nn.BatchNorm2d(out_channels),
        # norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )


# Pyramid scene pooling
class PSPPooling(nn.Module):
    def __init__(self, in_channels):
        super(PSPPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels)

        self.conv_final = nn.Conv2d(in_channels * 2,in_channels,1)

    def forward(self, x):
        size = x.size()[2:]
        # print(f'input size {size}')
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        
        # print(f'feat1 {feat1.size()}')
        # print(f'feat1 {feat2.size()}')
        # print(f'feat1 {feat3.size()}') 
        # print(f'feat1 {feat4.size()}')

        cat = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        output = self.conv_final(cat)
        # print(f' output size of psp pooling is {output.size()}')
        return output

####  ResUNet-a D6  Implmentation https://arxiv.org/pdf/1904.00592.pdf

class ResUNetA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResUNetA, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, 32, 1, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 1, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 1, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 1, stride=2)
        self.conv5 = nn.Conv2d(256, 512, 1, stride=2)
        self.conv6 = nn.Conv2d(512, 1024, 1, stride=2)
         
        
        self.convup1 = nn.Conv2d(1024, 512, 1, stride=1)
        self.convup2 = nn.Conv2d(512, 256, 1, stride=1)
        self.convup3 = nn.Conv2d(256, 128, 1, stride=1)
        self.convup4 = nn.Conv2d(128, 64, 1, stride=1)
        self.convup5 = nn.Conv2d(64, 32, 1, stride=1)

        self.conv_final = nn.Conv2d(32,1,1,stride=1)


        self.rest_block_1 = ResBlockAD4(32, 32, [1, 3, 15, 31],'down1')
        self.rest_block_2 = ResBlockAD4(64, 64, [1, 3, 15, 31],'down2')
        self.rest_block_3 = ResBlockAD3(128, 128, [1, 3, 15])
        self.rest_block_4 = ResBlockAD3(256, 256, [1, 3, 15])
        self.rest_block_5 = ResBlockAD1(512, 512, [1])
        self.rest_block_6 = ResBlockAD1(1024, 1024, [1])

        self.rest_block_up_1 = ResBlockAD4(32, 32, [1, 3, 15, 31],'up1')
        self.rest_block_up_2 = ResBlockAD4(64, 64, [1, 3, 15, 31],'up2')
        self.rest_block_up_3 = ResBlockAD3(128, 128, [1, 3, 15])
        self.rest_block_up_4 = ResBlockAD3(256, 256, [1, 3, 15])
        self.rest_block_up_5 = ResBlockAD1(512, 512, [1])

        self.PSPPooling = PSPPooling(1024)
        self.PSPPoolingResult = PSPPooling(32)


        self.combine1 = Combine(512)
        self.combine2 = Combine(256)
        self.combine3 = Combine(128)
        self.combine4 = Combine(64)
        self.combine5 = Combine(32)



    def forward(self, x):

        #  Encoder part with  6 Resblock-a  (D6)
        x_conv_1 = self.conv1(x)
        # print(x_conv_1.size())
        x_resblock_1 = self.rest_block_1(x_conv_1)
        
        x_conv_2 = self.conv2(x_resblock_1)

        # print(f'conv2 size: {x_conv_2.size()}')
        x_resblock_2 = self.rest_block_2(x_conv_2)

        x_conv_3 = self.conv3(x_resblock_2)
        # print(f'conv3 size: {x_conv_3.size()}')
        x_resblock_3 = self.rest_block_3(x_conv_3)

        x_conv_4 = self.conv4(x_resblock_3)
        # print(f'conv4 size: {x_conv_4.size()}')
        x_resblock_4 = self.rest_block_4(x_conv_4)

        x_conv_5 = self.conv5(x_resblock_4)
        # print(f'conv5 size: {x_conv_4.size()}')
        x_resblock_5 = self.rest_block_5(x_conv_5)

        x_conv_6 = self.conv6(x_resblock_5)
        # print(f'conv6 size: {x_conv_5.size()}')
        x_resblock_6 = self.rest_block_6(x_conv_6)

        x_pooling_1 = self.PSPPooling(x_resblock_6)
        
        #Decoder part  upsampling with combine
       
       #up5

        # print(f'pooling 1 size : {x_pooling_1.size()}')
        x_convup_1 = self.convup1(x_pooling_1)
        # x_upsampling1 = nn.UpsamplingNearest2d(scale_factor=2)(x_convup_1)
        x_upsampling1 = nn.UpsamplingNearest2d(scale_factor=2)(x_convup_1)

        # print(f'upsampling 1 size: {x_upsampling1.size()}  x_restblock_5 size: {x_resblock_5.size()}')
        x_combine_1 = self.combine1(x_upsampling1, x_resblock_5)
        # print(f'combine 1 output {x_combine_1.size()}')
        x_resblockup_5 = self.rest_block_up_5(x_combine_1)

      #up4
        # print(f'resblockup 5  size : {x_resblockup_5.size()}')
        x_convup_2 = self.convup2(x_resblockup_5)
        # print(f'convup2  size : {x_convup_2.size()}')
        x_upsampling2 = nn.UpsamplingNearest2d(scale_factor=2)(x_convup_2)
        
        # print(f'x_upsampling2 size : {x_upsampling2.size()} x_resblock_4 size: {x_resblock_4.size()}')
        x_combine_2 = self.combine2(x_upsampling2, x_resblock_4)
        x_resblockup_4 = self.rest_block_up_4(x_combine_2)



      #up3

        x_convup_3 = self.convup3(x_resblockup_4)
        x_upsampling3 = nn.UpsamplingNearest2d(scale_factor=2)(x_convup_3)
        x_combine_3 = self.combine3(x_upsampling3, x_resblock_3)
        x_resblockup_3 = self.rest_block_up_3(x_combine_3)

     #up2
        x_convup_4 = self.convup4(x_resblockup_3)
        x_upsampling4 = nn.UpsamplingNearest2d(scale_factor=2)(x_convup_4)
        x_combine_4 = self.combine4(x_upsampling4, x_resblock_2)
        x_resblockup_2 = self.rest_block_up_2(x_combine_4)


      #up1
        x_convup_5 = self.convup5(x_resblockup_2)
        x_upsampling5 = nn.UpsamplingNearest2d(scale_factor=2)(x_convup_5)
        x_combine_5 = self.combine5(x_upsampling5, x_resblock_1)
        x_resblockup_1 = self.rest_block_up_1(x_combine_5)

        x_combine_6 = self.combine5(x_resblockup_1, x_conv_1)
        
        # print(f'x_combine6 size: {x_combine_6.size()}')
        x_pooling_2 = self.PSPPoolingResult(x_combine_6)

        x_conv_result = self.conv_final(x_pooling_2)

        return x_conv_result


class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch,
                               mid_ch,
                               kernel_size=3,
                               padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch,
                               out_ch,
                               kernel_size=3,
                               padding=1,
                               bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

