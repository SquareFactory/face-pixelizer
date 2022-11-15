"""Copyright (C) SquareFactory SA - All Rights Reserved.
This source code is protected under international copyright law. All rights 
reserved and protected by the copyright holders.
This file is confidential and only available to authorized individuals with the
permission of the copyright holders. If you encounter this file and do not have
permission, please contact the copyright holders and delete this file.
"""
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from albumentations.core.serialization import from_dict
from retinaface.box_utils import decode
from retinaface.data_augment import Preproc
from retinaface.dataset import FaceDetectionDataset, detection_collate
from retinaface.multibox_loss import MultiBoxLoss
from retinaface.prior_box import priorbox
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import nms

from retina import clip_all_boxes, retinaface

TRAIN_IMAGE_PATH = Path("data/WIDER_train/WIDER_train/images")
VAL_IMAGE_PATH = Path("data/WIDER_val/WIDER_val/images")
# TEST_IMAGE_PATH = Path("data/WIDER_test/WIDER_test/images")

TRAIN_LABEL_PATH = Path("data/WIDER_labels/annotations/train/label.json")
VAL_LABEL_PATH = Path("data/WIDER_labels/annotations/val/label.json")


class RetinaFace(pl.LightningModule):
    def __init__(self, config):
        """doc here
        not up to date
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.config = config
        self.model = retinaface(config)
        self.image_size = tuple(config["image_size"])

        self.loss_factors = config["loss_factors"]

        self.priors = priorbox(
            image_size=self.image_size,
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False,
        )

        self.loss = MultiBoxLoss(
            num_classes=2,
            overlap_thresh=0.35,
            prior_for_matching=True,
            bkg_label=0,
            neg_mining=True,
            neg_pos=config["neg_pos_ratio"],
            neg_overlap=0.35,
            encode_target=False,
            priors=self.priors,
        )

        self.mAP = MeanAveragePrecision()

        self.mAP_epoch_interval = config[
            "mAP_epoch_interval"
        ]  # mAP takes a lot of time to compute, so better to not dot it everytime

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        return self.model(batch)

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
            momentum=self.config["momentum"],
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            T_0=self.config["scheduler"]["T_0"],
            T_mult=self.config["scheduler"]["T_mult"],
            optimizer=optimizer,
        )
        return [optimizer], [scheduler]  # type: ignore

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):  # type: ignore
        images = batch["image"]
        targets = batch["annotation"]
        out = self.forward(images)

        loss_localization, loss_classification, loss_landmarks = self.loss(out, targets)

        total_loss = (
            self.loss_factors["loc"] * loss_localization
            + self.loss_factors["cls"] * loss_classification
            + self.loss_factors["ldm"] * loss_landmarks
        )

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):  # type: ignore
        images = batch["image"]
        targets = batch["annotation"]
        batch["file_name"]

        for t in targets:
            t[:, 14] = -1
        image_height = images.shape[2]
        image_width = images.shape[3]
        out = self.forward(images)

        loss_localization, loss_classification, _ = self.loss(out, targets)

        total_loss = (
            self.loss_factors["loc"] * loss_localization
            + self.loss_factors["cls"] * loss_classification
        )

        # Computing mAP to track progress
        location, confidence, _ = out
        confidence = F.softmax(confidence, dim=-1)
        batch_size = location.shape[0]

        predictions: List[Dict[str, Any]] = []
        gtruth: List[Dict[str, Any]] = []

        scale = torch.from_numpy(np.tile([image_height, image_width], 2)).to(
            location.device
        )
        priors_cuda = self.priors.to(images.device)

        for batch_id in range(batch_size):
            boxes = decode(
                location.data[batch_id], priors_cuda, self.config["variance"]
            )
            scores = confidence[batch_id][:, 1]
            valid_index = torch.where(scores > self.config["detection_thres_for_mAP"])[
                0
            ]

            boxes = boxes[valid_index]
            scores = scores[valid_index]
            boxes *= scale
            boxes = clip_all_boxes(
                boxes, (image_height, image_width), in_place=False
            ).to(images.device)

            # do NMS
            keep = nms(boxes, scores, self.config["nms_threshold"])
            boxes = boxes[keep, :]
            if boxes.shape[0] == 0:
                continue

            scores = scores[keep]

            target_boxes = targets[batch_id][:, :4] * scale

            predictions.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": torch.ones(
                        len(scores), dtype=torch.int, device=images.device
                    ),
                }
            )

            gtruth.append(
                {
                    "boxes": target_boxes,
                    "labels": torch.ones(
                        len(target_boxes), dtype=torch.int, device=images.device
                    ),
                }
            )
        if (self.current_epoch + 1) % self.mAP_epoch_interval == 0:
            self.mAP.update(predictions, gtruth)
        return total_loss.cpu().detach()

    def validation_epoch_end(self, outputs: List) -> None:
        mean_val_loss = np.mean(outputs)
        self.log(
            "val_loss",
            mean_val_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        if (self.current_epoch + 1) % self.mAP_epoch_interval == 0:

            start = time.time()
            all_map_value = self.mAP.compute()
            end = time.time()

            all_map_value["computation_time"] = end - start
            map_value = all_map_value["map"]

            print(f"Last mAP calculated in {end-start:.2f} seconds")

            self.log(
                "map",
                map_value,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )


class FaceDataModule(pl.LightningDataModule):
    """doc here"""

    def __init__(
        self,
        config,
        train_data_dir=TRAIN_IMAGE_PATH,
        val_data_dir=VAL_IMAGE_PATH,
        train_labels_path=TRAIN_LABEL_PATH,
        val_labels_path=VAL_LABEL_PATH,
    ):

        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir

        self.train_labels_path = train_labels_path
        self.val_labels_path = val_labels_path

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.image_size = tuple(config["image_size"])
        self.aug_cfg = config["augmentations"]

    def setup(self, stage=None) -> None:  # type: ignore
        # check if dataset exists and download it. Not anymore: handled by docker build
        # train_ok = TRAIN_IMAGE_PATH.exists()
        # val_ok = VAL_IMAGE_PATH.exists()
        # lbl_ok = TRAIN_LABEL_PATH.exists() and VAL_LABEL_PATH.exists()

        # download_data(not train_ok, not val_ok, labels=not lbl_ok, unzip=True)
        self.preproc = Preproc(img_dim=self.image_size[0])

    def train_dataloader(self):
        result = DataLoader(
            FaceDetectionDataset(
                label_path=TRAIN_LABEL_PATH,
                image_path=TRAIN_IMAGE_PATH,
                transform=from_dict(self.aug_cfg["train_aug"]),
                preproc=self.preproc,
                rotate90=False,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=detection_collate,
            persistent_workers=True,
        )

        return result

    def val_dataloader(self):
        result = DataLoader(
            FaceDetectionDataset(
                label_path=VAL_LABEL_PATH,
                image_path=VAL_IMAGE_PATH,
                transform=from_dict(self.aug_cfg["val_aug"]),
                preproc=self.preproc,
                rotate90=False,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=detection_collate,
            persistent_workers=True,
        )
        return result

    # Not implemented yet
    # def test_dataloader(self):
    #     result = DataLoader(
    #         FaceDetectionDataset(
    #             label_path=TEST_LABEL_PATH,
    #             image_path=TEST_IMAGE_PATH,
    #             transform=from_dict(self.aug_cfg["test_aug"]),
    #             preproc=self.preproc,
    #             rotate90=False,
    #         ),
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=True,
    #         drop_last=True,
    #         collate_fn=detection_collate,
    #         persistent_workers=True,
    #     )
    #     return result


def main() -> None:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg(
        "-w",
        "--weights_path",
        type=str,
        help="path to the networks weights",
        default=None,
    )
    arg("-e", "--epochs", type=int, help="the number of epochs", default=10)
    arg(
        "-c",
        "--config",
        type=str,
        help="path to config yml",
        default="retina/config.yml",
    )

    args = parser.parse_args()

    cfg = Path(args.config)
    with cfg.open() as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # add to config the path given in args
    config["weights_path"] = args.weights_path

    pl.trainer.seed_everything(config["seed"])

    pipeline = RetinaFace(config)
    gpu_count = torch.cuda.device_count()

    dm = FaceDataModule(config)

    trainer = pl.Trainer(
        accelerator="gpu" if gpu_count > 0 else "cpu",
        gpus=gpu_count,
        max_epochs=args.epochs,
        num_sanity_val_steps=1,
        progress_bar_refresh_rate=1,
        benchmark=True,
        precision=16 if gpu_count > 0 else 32,
        sync_batchnorm=True,
        callbacks=pl.callbacks.ModelCheckpoint(**config["checkpoint_callback"]),
    )
    trainer.fit(pipeline, dm)


if __name__ == "__main__":
    main()
