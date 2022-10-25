from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2

from models.custom_retinaface import retinaface
from utils.data_augmentation import Preproc
from utils.data_download import download_data
from utils.multibox_loss import MultiBoxLoss
from utils.utils import get_prior_box

TRAIN_IMAGE_PATH = Path("WIDER_train/WIDER_train/images")
VAL_IMAGE_PATH = Path("WIDER_val/WIDER_val/images")

TRAIN_LABEL_PATH = Path("annotations/train/label.json")
VAL_LABEL_PATH = Path("annotations/val/label.json")


class RetinaFace(pl.LightningModule):
    def __init__(self, config):
        """doc here
        not up to date
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.config = config  # <TODO> see for what it is useful // maybe just pass the path instead of the whole dict?
        self.model = retinaface(config)
        self.image_size = config["image_size"]

        self.loss_factors = config["loss_factors"]

        self.priors = get_prior_box(
            height=self.image_size[0],
            width=self.image_size[1],
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
            neg_pos=7,
            neg_overlap=0.35,
            encode_target=False,
            priors=self.priors,
        )

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        return self.model(batch)


class FaceDataModule(pl.LightningDataModule):
    """doc here"""

    # TODO improve default values mgmt
    def __init__(
        self,
        config,
        train_data_dir=TRAIN_IMAGE_PATH,
        val_data_dir=VAL_IMAGE_PATH,
        train_labels_path=TRAIN_LABEL_PATH,
        val_labels_path=VAL_LABEL_PATH,
        batch_size=256,
        num_workers=2,
    ):

        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir

        self.train_labels_path = train_labels_path
        self.val_labels_path = val_labels_path

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.image_size = config["image_size"]

        input_size = max(self.image_size)
        quick_means = [104.0, 117.0, 123.0]  # TODO deal with this ugly impl.

        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size=input_size),
                A.PadIfNeeded(
                    min_height=input_size,
                    min_width=input_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Normalize(mean=quick_means, std=[1, 1, 1]),
                ToTensorV2(),
            ]
        )

    def setup(self, stage=None) -> None:  # type: ignore
        # check if dataset exists
        train_ok = TRAIN_IMAGE_PATH.exists()
        val_ok = VAL_IMAGE_PATH.exists()
        download_data(train_ok, val_ok, unzip=True)
        self.preproc = Preproc(img_dim=self.config.image_size[0])
