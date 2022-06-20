import pytorch_lightning as pl
import torch
from torch import nn
import torchvision.models as models
import timm
from data_utils import WiderFaceDataModule


class FaceDetectorModel(pl.LightningModule):
    def __init__(self, model_name="mobilenetv2_050", num_classes=28) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes, in_chans=3
        )

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
        return torch.optim.Adam(self.parameters(), lr=1e-3)

if __name__ == "__main__":
    dm = WiderFaceDataModule()
    dm.prepare_data()
    dm.setup()
    tr = iter(dm.train_dataloader())
    model = FaceDetectorModel()
    trainer = pl.Trainer()
    trainer.fit(model, dm)