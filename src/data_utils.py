from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms as T
from datasets import load_dataset


class WiderFaceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = T.Compose([T.ToTensor()])
        # maybe also normelize

    def setup(self, stage: Optional[str] = None):
        # download the dataset
        # to do: only load train and validation sets
        self.dataset = load_dataset("wider_face", data_dir=self.data_dir)
        # Assign train/val datasets for use in dataloaders
        self.train_set = self.dataset["train"]
        self.train_set.with_transform(self.transforms)
        self.val_set = self.dataset["validation"]
        self.val_set.with_transform(self.transforms)


    def prepare_data(self):
        load_dataset("wider_face", data_dir=self.data_dir)
        
    def transforms(self, data):
        #  [self.transform(image["image"]) for image in data]
        new_data = []
        for example in data:
            example["image"] = self.transform(example["image"])
            new_data.append(example)
        return new_data

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=32)


dm = WiderFaceDataModule()
dm.prepare_data()
dm.setup()
print(next(iter(dm.train_dataloader())))
