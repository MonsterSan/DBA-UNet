# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneckV3(nn.Module):
    expansion = 4

    def __init__(self, cin, cout=None, cmid=None, stride=1, attention=None, downsample=None):
        super(PreActBottleneckV3, self).__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        # print("attention mode:\t"+attention)

        self.conv1 = conv1x1(cin, cmid, bias=False)
        # self.bn1 = nn.BatchNorm2d(cmid)
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        # self.bn2 = nn.BatchNorm2d(cmid)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        # self.bn3 = nn.BatchNorm2d(cout)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.gn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            residual = self.gn_proj(residual)
        out += residual
        out = self.relu(out)

        return out


class AttentionResNet(nn.Module):
    def __init__(self, block_units=(3, 4, 6, 3), width_factor=1,
                 resnet_attention=None,
                 aspp=None,
                 attention_block=None):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneckV3(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneckV3(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneckV3(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneckV3(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneckV3(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneckV3(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneckV3(cin=width * 16, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneckV3(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[3] + 1)],
            ))),
        ]))

    def forward(self, x):
        # print("resnet input size:{}".format(str(x.shape)))
        output = {}

        b, c, in_size, _ = x.size()
        x = self.root(x)  # x1
        output["x1"] = x
        # print("x1 shape {}".format(x.shape))
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        # print("x shape after maxpool {}".format(x.shape))

        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            if i == 0:
                output["x2"] = x
                # print("x2 shape {}".format(x.shape))
            if i == 1:
                output["x3"] = x
                # print("x3 shape {}".format(x.shape))
            if i == 2:
                output["x4"] = x
                # print("x4 shape {}".format(x.shape))

            # right_size = int(in_size / 4 / (i + 1))
            ##if x.size()[2] != right_size:
            #    pad = right_size - x.size()[2]
            #    assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
            #    feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
            #    feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            # else:
            #    feat = x
        x = self.body[-1](x)
        output["x5"] = x
        # print("x5 shape {}".format(x.shape))
        # print("resnet output size:{}".format(str(x.shape)))
        return output
        # x1 32*32*x.H/2,  x.W/2
        # x2 32*32*x.H/4,  x.W/4


resnet = AttentionResNet()


class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)   (N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)  (N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)    (N, 32, x.H/2,  x.W/2)
        # print("x1 shape   {}".format(x1.shape))
        # print("x2 shape   {}".format(x2.shape))
        # print("x3 shape   {}".format(x3.shape))
        # print("x4 shape   {}".format(x4.shape))
        # print("x5 shape   {}".format(x5.shape))
        score = self.bn1(self.relu(self.deconv1(x5)))  # size=(N, 512, x.H/16, x.W/16)
        score = score + x4  # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3  # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2  # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1  # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 16, x.H, x.W)
        score = self.classifier(score)  # size=(16, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)


def FCN(num_classes):
    return FCNs(n_class=num_classes, pretrained_net=resnet)


if __name__ == '__main__':
    model = FCN(2)
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    print(out.shape)