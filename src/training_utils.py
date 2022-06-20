import pytorch_lightning as pl
import torch
from torch import nn
import torchvision.models as models
import timm
from data_utils import WiderFaceDataModule
from argparse import ArgumentParser


class FaceDetectorModel(pl.LightningModule):
    def __init__(
        self, model_name="mobilenetv2_050", num_classes=28, learning_rate=1e-4
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes, in_chans=3
        )
        self.learning_rate = learning_rate

    @staticmethod
    def add_argparse_args(parent_parser):
        """Argument parser for model."""
        parser = parent_parser.add_argument_group("Classifier")
        parser.add_argument("--learning_rate", type=float, default=0.0005)
        return parent_parser

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
        images, faces = batch["image"], batch["faces"]
        output = self(images)
        loss = nn.functional.mse_loss(output, faces.reshape(faces.shape[0], -1))
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
        images, faces = batch["image"], batch["faces"]
        output = self(images)
        loss = nn.functional.mse_loss(output, faces.reshape(faces.shape[0], -1))
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        """retrieving the optimizer.
        Args:
            None.
        Returns:
            optimizer: the optimizer.
        Raises:
            None.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = WiderFaceDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)

    dm = WiderFaceDataModule(args)
    dm.prepare_data()
    dm.setup()
    model = FaceDetectorModel()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)
