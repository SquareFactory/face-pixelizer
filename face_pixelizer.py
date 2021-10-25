import argparse
import copy
import os
import time
from typing import List
import warnings

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from retinaface import retinaface
from utils import decode_boxes, get_prior_box, pixelize


warnings.simplefilter("ignore")


class FacePixelizer:
    def __init__(
        self,
        input_size: int = 512,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        state_dict: str = "/opt/face_pixelizer/retinaface_mobilenet_0.25.pth",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        height, width = input_size, input_size

        self.model = retinaface(state_dict)
        self.model.eval()
        dump_inputs = torch.randn(1, 3, height, width)
        self.model = torch.jit.trace(self.model, dump_inputs)
        self.model = self.model.to(self.device)

        self.tf = A.Compose(
            [
                A.LongestMaxSize(max_size=input_size),
                A.PadIfNeeded(
                    min_height=height,
                    min_width=width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
            ]
        )

        self.priors = get_prior_box(height, width).to(self.device)
        self.boxes_scale = torch.Tensor([width, height] * 2).to(self.device)

        print(f"Face pixelizer setup! (on {self.device})")

    def __call__(self, imgs: List[np.ndarray]) -> List[np.ndarray]:

        # Be sure we not modify inputs
        imgs = copy.copy(imgs)

        # transforms imgs to tensors

        tensors = []
        for img in imgs:
            img = img.astype(np.float32) - (104, 117, 123)
            img = self.tf(image=img)["image"]
            tensor = torch.from_numpy(img).permute(2, 0, 1)
            tensors.append(tensor.unsqueeze(0))
        tensors = torch.cat(tensors).type(torch.FloatTensor).to(self.device)

        # Inferences

        with torch.no_grad():
            boxes, scores = self.model(tensors)

        # Analyze outputs

        variances = [0.1, 0.2]
        boxes = decode_boxes(boxes, self.priors, variances)
        boxes = boxes * self.boxes_scale
        scores = scores[:, :, 1]

        processed_imgs = []
        for img, boxes_per_img, scores_per_img in zip(imgs, boxes, scores):
            # Remove low scores
            inds = torch.gt(scores_per_img, self.score_threshold)
            boxes_per_img = boxes_per_img[inds]
            scores_per_img = scores_per_img[inds]

            # NMS
            keep = torchvision.ops.boxes.nms(
                boxes_per_img, scores_per_img, self.nms_threshold
            )
            scores_per_img = scores_per_img[keep]
            boxes_per_img = boxes_per_img[keep]

            # Deaugmente results
            original_shape = img.shape[:2]
            scale = self.input_size / max(original_shape)
            padding = int((self.input_size - min(original_shape) * scale) / 2)
            for box in boxes_per_img:
                # Remove padding
                start_coord = 0 if np.argmax(original_shape) == 0 else 1
                box[start_coord] -= padding
                box[start_coord + 2] -= padding
                # Remove scale
                box = box / scale
                box = box.type(torch.int)

                # Apply pixelization on faces
                img[box[1] : box[3], box[0] : box[2]] = pixelize(
                    img[box[1] : box[3], box[0] : box[2]]
                )
            processed_imgs.append(img)

        return processed_imgs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"{args.image_path} do not exist")

    img = cv2.imread(args.image_path)
    if img is None:
        raise ValueError(f"{args.image_path} is invalid")
    input_shape = img.shape

    # Setup model

    face_pixelizer = FacePixelizer(
        input_size=512, state_dict="retinaface_mobilenet_0.25.pth"
    )

    # Inference

    start = time.time()
    pred = face_pixelizer([img])[0]
    print(f"inference done in {time.time() - start:0.3f} secs. {input_shape}")

    # Plot images

    f, axes = plt.subplots(2, 1)
    for axe, im, title in zip(axes, [img, pred], ["original", "prediction"]):
        axe.imshow(im[..., ::-1])
        axe.set_title(title)
        axe.axis("off")
    plt.tight_layout()
    plt.show()
