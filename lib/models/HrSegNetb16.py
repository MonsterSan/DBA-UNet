import math

import torch
import torch.nn as nn

import torch.nn.functional as F


class HrSegNetB16(nn.Module):
    """
    The HrSegNet implementation based on PaddlePaddle.s

    Args:
        num_classes (int): The unique number of target classes.

        in_channels (int, optional): The channels of input image. Default: 3.

        base (int, optional): The base channel number of the model. Default: 16.
    """

    def __init__(self,
                 in_channels=3,  # input channel
                 base=16,  # base channel of the model,
                 num_classes=2,  # number of classes
                 pretrained=None  # pretrained model
                 ):
        super(HrSegNetB16, self).__init__()
        self.base = base
        self.num_classed = num_classes
        self.pretrained = pretrained
        # Stage 1 and 2 constitute the stem of the model, which is mainly used to extract low-level features.
        # Meanwhile, stage1 and 2 reduce the input image to 1/2 and 1/4 of the original size respectively
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=base // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base // 2),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=base // 2, out_channels=base, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(),
        )

        self.seg1 = SegBlock(base=base, stage_index=1)
        self.seg2 = SegBlock(base=base, stage_index=2)
        self.seg3 = SegBlock(base=base, stage_index=3)

        self.aux_head1 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.aux_head2 = SegHead(inplanes=base, interplanes=base, outplanes=num_classes, aux_head=True)
        self.head = SegHead(inplanes=base, interplanes=base, outplanes=num_classes)

    def forward(self, x):
        logit_list = []
        h, w = x.shape[2:]
        # aux_head only used in training
        stem1_out = self.stage1(x)
        stem2_out = self.stage2(stem1_out)
        hrseg1_out = self.seg1(stem2_out)
        hrseg2_out = self.seg2(hrseg1_out)
        hrseg3_out = self.seg3(hrseg2_out)
        last_out = self.head(hrseg3_out)
        seghead1_out = self.aux_head1(hrseg1_out)
        seghead2_out = self.aux_head2(hrseg2_out)
        last_out = F.interpolate(last_out, size=(h, w), mode='bilinear', align_corners=True)
        seghead1_out = F.interpolate(seghead1_out, size=(h, w), mode='bilinear', align_corners=True)
        seghead2_out = F.interpolate(seghead2_out, size=(h, w), mode='bilinear', align_corners=True)
        return last_out, seghead1_out, seghead2_out


class SegBlock(nn.Module):
    def __init__(self,
                 base=32,
                 stage_index=1):  # stage_index=1,2,3.
        super(SegBlock, self).__init__()

        #  Convolutional layer for high-resolution paths with constant spatial resolution and constant channel
        self.h_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=base, out_channels=base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU()
        )
        self.h_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=base, out_channels=base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU()
        )
        self.h_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=base, out_channels=base, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU()
        )

        # sematic guidance path/low-resolution path
        if stage_index == 1:  # first stage, stride=2, spatial resolution/2, channel*2
            self.l_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=base, out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=2,
                          padding=1),
                nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
                nn.ReLU()
            )
        elif stage_index == 2:  # second stage
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=base, out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=2,
                          padding=1),
                nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
                nn.ReLU()
            )
        elif stage_index == 3:
            self.l_conv1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=base, out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=2,
                          padding=1),
                nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
                nn.ReLU(),
                nn.Conv2d(in_channels=base * int(math.pow(2, stage_index)),
                          out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
                nn.ReLU()
            )
        else:
            raise ValueError("stage_index must be 1, 2 or 3")
        self.l_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=base * int(math.pow(2, stage_index)),
                      out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
            nn.ReLU()
        )
        self.l_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=base * int(math.pow(2, stage_index)),
                      out_channels=base * int(math.pow(2, stage_index)), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base * int(math.pow(2, stage_index))),
            nn.ReLU()
        )

        self.l2h_conv1 = nn.Conv2d(in_channels=base * int(math.pow(2, stage_index)), out_channels=base, kernel_size=1,
                                   stride=1, padding=0)
        self.l2h_conv2 = nn.Conv2d(in_channels=base * int(math.pow(2, stage_index)), out_channels=base, kernel_size=1,
                                   stride=1, padding=0)
        self.l2h_conv3 = nn.Conv2d(in_channels=base * int(math.pow(2, stage_index)), out_channels=base, kernel_size=1,
                                   stride=1, padding=0)

    def forward(self, x):
        # out = []
        # out.append(self.h_conv3(self.h_conv2(self.h_conv1(x))))
        # out.append(self.l_conv3(self.l_conv2(self.l_conv1(x))))
        size = x.shape[2:]
        out_h1 = self.h_conv1(x)  # high resolution path
        out_l1 = self.l_conv1(x)  # low resolution path
        # print(out_l1.shape)
        out_l1_i = F.interpolate(out_l1, size=size, mode='bilinear', align_corners=True)  # upsample
        out_hl1 = self.l2h_conv1(out_l1_i) + out_h1  # low to high

        out_h2 = self.h_conv2(out_hl1)
        out_l2 = self.l_conv2(out_l1)
        # print(out_l2.shape)
        out_l2_i = F.interpolate(out_l2, size=size, mode='bilinear', align_corners=True)
        out_hl2 = self.l2h_conv2(out_l2_i) + out_h2

        out_h3 = self.h_conv3(out_hl2)
        out_l3 = self.l_conv3(out_l2)
        # print(out_l3.shape)
        out_l3_i = F.interpolate(out_l3, size=size, mode='bilinear', align_corners=True)
        out_hl3 = self.l2h_conv3(out_l3_i) + out_h3
        return out_hl3


# seg head
class SegHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, aux_head=False):
        super(SegHead, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        if aux_head:
            self.con_bn_relu = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=interplanes, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(interplanes),
                nn.ReLU(),
            )
        else:
            self.con_bn_relu = nn.Sequential(
                nn.ConvTranspose2d(in_channels=inplanes, out_channels=interplanes, kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(interplanes),
                nn.ReLU(),
            )
        self.conv = nn.Conv2d(in_channels=interplanes, out_channels=outplanes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.con_bn_relu(x)
        out = self.conv(x)
        return out


if __name__ == "__main__":
    model = HrSegNetB16()
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
