from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms

from models.custom_retinaface import retinaface
from utils.data_augmentation import Preproc
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

    def __init__(
        self,
        train_data_dir=TRAIN_IMAGE_PATH,
        val_data_dir=VAL_IMAGE_PATH,
        train_labels_path=TRAIN_LABEL_PATH,
        val_labels_path=VAL_LABEL_PATH,
        batch_size=256,
        num_workers=2,
    ):
        # TODO improve default values mgmt
        super().__init__()
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir

        self.train_labels_path = train_labels_path
        self.val_labels_path = val_labels_path

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def setup(self, stage=0) -> None:  # type: ignore
        self.preproc = Preproc(
            img_dim=self.config.image_size[0]
        )  # Is it necessary in setup? could it be called at init?
