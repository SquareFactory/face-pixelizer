from typing import Dict
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ..models.base_nets import FPN, SSH, MobileNetV1, BboxHead, ClassHead, LandmarkHead

class RetinaFace_no_landm(nn.Module):
    def __init__(self):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace_no_landm, self).__init__()

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

class RetinaFace(nn.Module):
    def __init__(self):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()

        backbone = MobileNetV1()
        if (Path("MobileNetv1.pth").exists()):
            backbone.load_state_dict(torch.load("MobileNetv1.pth"))
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


def retinaface(config: Dict = {}, landmarks = True):
    if landmarks:
        model = RetinaFace()
    else:
        model = RetinaFace_no_landm()

    if "weights_path" in config:
        weights_path = config["weights_path"]
        if weights_path is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            weights = torch.load(weights_path, map_location=torch.device(device))
            model.load_state_dict(weights)

    return model
