"""Copyright (C) SquareFactory SA - All Rights Reserved.
This source code is protected under international copyright law. All rights 
reserved and protected by the copyright holders.
This file is confidential and only available to authorized individuals with the
permission of the copyright holders. If you encounter this file and do not have
permission, please contact the copyright holders and delete this file.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ..models.base_nets import FPN, SSH, BboxHead, ClassHead, LandmarkHead, MobileNetV1


class RetinaFace_no_landm(nn.Module):
    def __init__(self):
        """
        RetinaFace network with a MobileNetV1 Backbone for high-speed inference.
        This version does not learn to predict facial landmarks location.

        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace_no_landm, self).__init__()

        self.backbone = MobileNetV1()
        return_layers = {"stage1": 1, "stage2": 2, "stage3": 3}
        in_channels = [32 * 2, 32 * 4, 32 * 8]
        out_channels = 64
        fpn_num = 3
        anchor_num = 2

        self.body = models._utils.IntermediateLayerGetter(self.backbone, return_layers)
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


class RetinaFace(nn.Module):
    def __init__(self):
        """
        RetinaFace network with a MobileNetV1 Backbone for high-speed inference.
        This version does not learn to predict facial landmarks location.
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()

        self.backbone = MobileNetV1()
        return_layers = {"stage1": 1, "stage2": 2, "stage3": 3}
        in_channels = [32 * 2, 32 * 4, 32 * 8]
        out_channels = 64
        fpn_num = 3
        anchor_num = 2

        self.body = models._utils.IntermediateLayerGetter(self.backbone, return_layers)
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

        self.LandmarkHead = nn.ModuleList()
        for _ in range(fpn_num):
            self.LandmarkHead.append(LandmarkHead(out_channels, anchor_num))

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

        ldm_regressions = [self.LandmarkHead[i](feat) for i, feat in enumerate(ssh)]
        ldm_regressions = torch.cat(ldm_regressions, dim=1)

        return bbox_regressions, classifications, ldm_regressions


def retinaface(config: Dict = {}, landmarks=True, untrained_backbone=False):
    """Easy access to any RetinaFace instance.
    MobileNet Backbone is pretrained by default.

    Args:
         config: dict which contains the path to weights, to get a trained network.
                 other key-values are ignored, except for 'backbone_weights_path'
                 which allows to get a pretrained backbone when the weights are not supplied"""
    if landmarks:
        model = RetinaFace()
    else:
        model = RetinaFace_no_landm()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "weights_path" in config and config["weights_path"] is not None:

        weights_path = config["weights_path"]
        weights = torch.load(weights_path, map_location=torch.device(device))
        model.load_state_dict(weights)
        print("Using pretrained weights for the retinaface network")

    elif (
        "backbone_weights_path" in config
        and config["backbone_weights_path"] is not None
    ):

        bb_w = torch.load(
            config["backbone_weights_path"], map_location=torch.device(device)
        )
        model.backbone.load_state_dict(bb_w)
        print("Using pretrained weights for the backbone only")
    else:
        print("Fully naive network")

    return model
