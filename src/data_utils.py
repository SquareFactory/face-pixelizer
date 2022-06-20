from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from datasets import load_dataset


class WiderFaceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = T.Compose([ T.Resize((256,256)), T.ToTensor()])
        # maybe also normelize

    def prepare_data(self):
        load_dataset("wider_face", data_dir=self.data_dir)

    def setup(self, stage: Optional[str] = None):
        self.dataset = load_dataset("wider_face")
        

        self.dataset.set_transform(self.pil_image_transform, columns=["image","faces"], output_all_columns=True)
        # max = 0
        # for i in range(len(self.dataset)):
        #     if len(self.dataset["train"]["faces"][i]) > max:
        #         max = len(self.dataset["train"]["faces"][i])
        # print(max)
        # self.dataset.set_transform(self.labels_transform, columns="faces")
        self.train_set = self.dataset["train"]
        self.val_set = self.dataset["validation"]



    def pil_image_transform(self, batch):
        return {"image": [self.transform(img) for img in batch["image"]],
        "faces": [label["bbox"][0] for label in batch["faces"]]
        } 

    def labels_transform(self, batch):
        return {"faces": [label["bbox"][0] for label in batch["faces"]]}
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=32)


dm = WiderFaceDataModule()
dm.prepare_data()
dm.setup()
print(dm.train_set.__getitem__(2))
print(next(iter(dm.train_dataloader())))
