from pathlib import Path
from typing import Tuple, Callable, Union, Any, Dict, List
import argparse
import albumentations as A
import cv2
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
#from albumentations.pytorch import ToTensorV2
from albumentations.core.serialization import from_dict
import yaml 
from torchvision.ops import nms
# from retina.utils.utils import clip_all_boxes
# from utils.data_augmentation import Preproc
# from utils.data_download import download_data
# from utils.multibox_loss import MultiBoxLoss
# from utils.utils import get_prior_box

from retinaface.box_utils import decode
from retinaface.data_augment import Preproc
from retinaface.dataset import FaceDetectionDataset, detection_collate
from retinaface.multibox_loss import MultiBoxLoss
from retinaface.prior_box import priorbox
from retina import download_data, retinaface, clip_all_boxes
#from ..models.custom_retinaface import retinaface

TRAIN_IMAGE_PATH = Path("data/WIDER_train/WIDER_train/images")
VAL_IMAGE_PATH = Path("data/WIDER_val/WIDER_val/images")

TRAIN_LABEL_PATH = Path("data/annotations/train/label.json")
VAL_LABEL_PATH = Path("data/annotations/val/label.json")


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
            neg_pos=7,
            neg_overlap=0.35,
            encode_target=False,
            priors=self.priors,
        )

        self.mAP = MeanAveragePrecision(iou_thresholds=[self.config["iou_threshold"]]).to('cuda')
        self.epoch_counter = 0

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        return self.model(batch)

    def configure_optimizers(self):
        
        optimizer = torch.optim.SGD(
              lr = self.config['lr'],
              weight_decay = self.config['weight_decay'],
              momentum = self.config['momentum'],
              params=[x for x in self.model.parameters() if x.requires_grad]
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            T_0 = self.config["scheduler"]["T_0"],
            T_mult = self.config["scheduler"]["T_mult"],
            optimizer=optimizer)
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
        
        self.log("train_classification", loss_classification, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log("train_localization", loss_localization, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log("train_landmarks", loss_landmarks, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        #self.log("lr", self._get_current_lr(), on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):  # type: ignore
        images = batch["image"]
        targets = batch["annotation"]
        file_names = batch["file_name"]


        image_height = images.shape[2]
        image_width = images.shape[3]
        # annotations = batch["annotation"]
        out = self.forward(images)
        loss_localization, loss_classification, _ = self.loss(out, targets)

        total_loss = (
            self.loss_factors["loc"] * loss_localization
            + self.loss_factors["cls"] * loss_classification
        )

        self.log("val_classification", loss_classification, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log("val_localization", loss_localization, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        #self.log("val_landmarks", loss_landmarks, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log("val_loss", total_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

        # Computing mAP to track progress
        location, confidence, _ = out

        confidence = F.softmax(confidence, dim=-1)
        batch_size = location.shape[0]
        

        predictions: List[Dict[str, Any]] = []
        gtruth : List[Dict[str, Any]] = []

        scale = torch.from_numpy(np.tile([image_width, image_height], 2)).to(location.device)
        priors_cuda = self.priors.to(images.device)

        for batch_id in range(batch_size):
            boxes = decode(
                location.data[batch_id], priors_cuda, self.config["variance"]
            )
            scores = confidence[batch_id][:, 1]

            valid_index = torch.where(scores > 0.1)[0]

            boxes = boxes[valid_index]
            scores = scores[valid_index]
            boxes *= scale
            boxes = clip_all_boxes(boxes,(image_height, image_width),in_place=False).to(images.device)

            # do NMS
            if self.global_step % 10000 == 0:
                print(f"Before NMS {len(boxes)}")
            
            keep = nms(boxes, scores, self.config["nms_threshold"])
            boxes = boxes[keep, :]
            if boxes.shape[0] == 0:
                continue
            
            if self.current_epoch % 10 == 0:
                print(f"After NMS {len(boxes)}")

            scores = scores[keep]
            target_boxes = targets[batch_id][:,:4]*scale
            predictions.append(
                {
                    "boxes":boxes,
                    "scores":scores,
                    "labels":torch.ones(len(scores), dtype=torch.int, device = images.device)
                }
            )

            
            gtruth.append(
                {
                    "boxes": target_boxes,
                    "labels" : torch.ones(len(target_boxes), dtype=torch.int, device = images.device)  
                }
            )
        self.mAP.update(predictions,gtruth)
        return total_loss
                # file_name = file_names[batch_id]

            # for box_id, bbox in enumerate(boxes):
            #     x_min, y_min, x_max, y_max = bbox

            #     x_min = np.clip(x_min, 0, x_max - 1)
            #     y_min = np.clip(y_min, 0, y_max - 1)

            #     predictions_coco += [
            #         {
            #             #"id": str(hash(f"{file_name}_{box_id}")),
            #             #"image_id": file_name,
            #             "category_id": 1,
            #             "bbox": [x_min, y_min, x_max, y_max],
            #             "score": scores[box_id],
            #         }
            #     ]

        # gt_coco: List[Dict[str, Any]] = []

        # for batch_id, annotation_list in enumerate(annotations):
        #     for annotation in annotation_list:
        #         x_min, y_min, x_max, y_max = annotation[:4]
        #         file_name = file_names[batch_id]

        #         gt_coco += [
        #             {
        #                 "id": str(hash(f"{file_name}_{batch_id}")),
        #                 "image_id": file_name,
        #                 "category_id": 1,
        #                 "bbox": [
        #                     x_min.item() * image_width,
        #                     y_min.item() * image_height,
        #                     (x_max - x_min).item() * image_width,
        #                     (y_max - y_min).item() * image_height,
        #                 ],
        #             }
        #         ]
        # return OrderedDict({"predictions": predictions_coco, "gt": gt_coco})

    def validation_epoch_end(self, outputs: List) -> None:
    
        if self.current_epoch > 60:
            print('*********************************************beforemap****************************************************')
            map_value = self.mAP.compute()
            self.log("map", map_value, on_step=False, on_epoch=True, logger=True)
            print('*********************************************aftermap****************************************************')
        else:
            self.mAP.reset()
        # result_predictions: List[dict] = []
        # result_gt: List[dict] = []

        # for output in outputs:
        #     result_predictions += output["predictions"]
        #     result_gt += output["gt"]

        # _, _, average_precision = recall_precision(result_gt, result_predictions, 0.5)

        

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
        self.image_size = config["image_size"]
        self.aug_cfg = config["aug_cfg"]
        self.num_workers = config["num_workers"]

        # input_size = max(self.image_size)
        # quick_means = config["mean_pix_val"]  # TODO deal with this ugly impl.

        # self.transform = A.Compose(
        #     [
        #         A.LongestMaxSize(max_size=input_size),
        #         A.PadIfNeeded(
        #             min_height=input_size,
        #             min_width=input_size,
        #             border_mode=cv2.BORDER_CONSTANT,
        #             value=0,
        #         ),
        #         A.Normalize(mean=quick_means, std=[1, 1, 1]),
        #         ToTensorV2(),
        #     ]
        # )

    def setup(self, stage=None) -> None:  # type: ignore
        # check if dataset exists
        train_ok = TRAIN_IMAGE_PATH.exists()
        val_ok = VAL_IMAGE_PATH.exists()
        lbl_ok =  TRAIN_LABEL_PATH.exists() and VAL_LABEL_PATH.exists()
        if not lbl_ok:
            raise ValueError("You can't download the labels yet") #TODO host json labels somewhere and change code in data_dld
        download_data(not train_ok, not val_ok, labels = not lbl_ok, unzip=True)
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
            persistent_workers=True
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


def main() -> None:
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    #arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-w", "--weights_path", type=str, help="path to the networks weights", default = None)
    arg("-e", "--epochs", type=int, help="the number of epoches", default=10)
    #arg("-b", "--batch_size", type=int, help="the batch size of the trainloaders", default=6)
    args =  parser.parse_args()


    aug_cfg = Path("retina/augmentations.yaml")
    with aug_cfg.open() as f:
        aug_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    config = {
        "weights_path": args.weights_path,
        "image_size" : (512,512),
        "loss_factors" : {"loc": 2, "cls" : 1, "ldm": 1},
        "lr" : 0.001,
        "weight_decay" : 0.0001,
        "momentum" : 0.9,
        "num_workers": 16,
        "mean_pix_val" : [104.0, 117.0, 123.0],
        "aug_cfg" : aug_config,
        "scheduler" : {"T_0": 10, "T_mult": 2},
        "batch_size" : 16,
        "variance": [0.1, 0.2],
        "iou_threshold" : 0.5,
        "nms_threshold" : 0.3,
        "seed" : 43
    }
    pl.trainer.seed_everything(config["seed"])

    pipeline = RetinaFace(config)
    gpu_count = torch.cuda.device_count()

    dm = FaceDataModule(config)

    
    trainer = pl.Trainer(
        accelerator= "gpu" if gpu_count > 0 else "cpu",
        gpus=gpu_count,
        max_epochs= args.epochs,
        num_sanity_val_steps= 1,
        progress_bar_refresh_rate= 1,
        benchmark= True,
        precision= 16 if gpu_count > 0 else 32,
        sync_batchnorm= True,
        #callbacks=[object_from_dict(config.checkpoint_callback)]
    )
    trainer.fit(pipeline, dm)

if __name__ == "__main__":
    main()
