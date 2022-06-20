from typing import Optional
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import load_dataset


class WiderFaceDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.dataloader_arguments = {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
        self.data_dir = args.data_dir
        self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        # maybe also normelize

    @staticmethod
    def add_argparse_args(parent_parser):
        """Argument parser for datamodule."""
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--data_dir", type=str, default="./")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=32)
        parser.add_argument("--seed", type=int, default=303)

        return parent_parser
    def prepare_data(self):
        load_dataset("wider_face", data_dir=self.data_dir)

    def setup(self, stage: Optional[str] = None):
        self.dataset = load_dataset("wider_face")

        self.dataset.set_transform(
            self.images_faces_transform,
            columns=["image", "faces"],
            output_all_columns=True,
        )

        self.train_set = self.dataset["train"]
        self.val_set = self.dataset["validation"]

    def images_faces_transform(self, batch):
        new_batch = {"image": torch.Tensor(), "faces": torch.Tensor()}
        for img, face in zip(batch["image"], batch["faces"]):
            new_batch["image"] = torch.cat((new_batch["image"], self.transform(img))).reshape(1,3,224,224)
            if len(face["bbox"]) <= 7:
                            bbox = torch.cat(
                                (
                                    torch.tensor(face["bbox"]),
                                    torch.zeros(7 - len(face["bbox"]), 4),
                                )
                            )
            else:
                bbox = torch.tensor(face["bbox"][:7])
            new_batch["faces"] = torch.cat((new_batch["faces"], bbox)).reshape(1,7,4)
        return new_batch

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=32)
