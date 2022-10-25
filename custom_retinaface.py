"""
mostly taken from https://github.com/biubug6/Pytorch_Retinaface

note that the landmarks part was removed since it is no usefull
in our case
"""


from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(
            out_channel // 4, out_channel // 4, stride=1, leaky=leaky
        )
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(
            in_channels_list[0], out_channels, stride=1, leaky=leaky
        )
        self.output2 = conv_bn1X1(
            in_channels_list[1], out_channels, stride=1, leaky=leaky
        )
        self.output3 = conv_bn1X1(
            in_channels_list[2], out_channels, stride=1, leaky=leaky
        )

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, inputs):
        # names = list(input.keys())
        inputs = list(inputs.values())

        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])

        up3 = F.interpolate(
            output3, size=[output2.size(2), output2.size(3)], mode="nearest"
        )
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(
            output2, size=[output1.size(2), output1.size(3)], mode="nearest"
        )
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return [output1, output2, output3]


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class ClassHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(
            in_channels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


# class LandmarkHead(nn.Module):
# def __init__(self, in_channels=512, num_anchors=3):
# super(LandmarkHead, self).__init__()
# self.conv1x1 = nn.Conv2d(
# in_channels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0
# )

# def forward(self, x):
# out = self.conv1x1(x)
# out = out.permute(0, 2, 3, 1).contiguous()
# return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):
    def __init__(self):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()

        backbone = MobileNetV1()
        return_layers = {"stage1": 1, "stage2": 2, "stage3": 3}
        in_channels = [32 * 2, 32 * 4, 32 * 8]
        out_channels = 64
        fpn_num = 3
        anchor_num = 2

        self.body = models._utils.IntermediateLayerGetter(backbone, return_layers)
        self.fpn = FPN(in_channels, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = nn.ModuleList()
        for _ in range(fpn_num):
            self.ClassHead.append(ClassHead(out_channels, anchor_num))

        self.BboxHead = nn.ModuleList()
        for _ in range(fpn_num):
            self.BboxHead.append(BboxHead(out_channels, anchor_num))

        # self.LandmarkHead = nn.ModuleList()
        # for _ in range(fpn_num):
        # self.LandmarkHead.append(LandmarkHead(out_channels, anchor_num))

    def forward(self, inputs):
        out = self.body(inputs)
        fpn = self.fpn(out)
        ssh = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]

        bbox_regressions = [self.BboxHead[i](feat) for i, feat in enumerate(ssh)]
        bbox_regressions = torch.cat(bbox_regressions, dim=1)

        classifications = [self.ClassHead[i](feat) for i, feat in enumerate(ssh)]
        classifications = torch.cat(classifications, dim=1)
        if not self.training:
            classifications = F.softmax(classifications, dim=-1)

        # ldm_regressions = [self.LandmarkHead[i](feat) for i, feat in enumerate(ssh)]
        # ldm_regressions = torch.cat(ldm_regressions, dim=1)

        # return bbox_regressions, classifications, ldm_regressions
        return bbox_regressions, classifications


def retinaface(config: Dict = {}):
    model = RetinaFace()
    if "weights_path" in config:
        weights_path = config["weights_path"]
        if weights_path is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            weights = torch.load(weights_path, map_location=torch.device(device))
            model.load_state_dict(weights)

    return model
