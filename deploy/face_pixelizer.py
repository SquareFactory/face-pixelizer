import argparse
import copy
import os
import time
import warnings
from typing import List

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

# If you did not set up the package w/ "pip install -e .", use paths instead:
from retina.models.custom_retinaface import retinaface
from retina.utils.utils import decode_boxes, get_prior_box, pixelize

# imports with package set up
# from retina import retinaface
# from retina import decode_boxes, get_prior_box, pixelize


warnings.simplefilter("ignore")


class FacePixelizer:
    def __init__(
        self,
        input_size: int = 512,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        state_dict_path: str = "/opt/face_pixelizer/retinaface_mobilenet_0.25.pth",
        use_landmarks=False,
        device="cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.use_landmarks = use_landmarks

        height, width = input_size, input_size

        self.model = retinaface(
            {"weights_path": state_dict_path}, use_landmarks=use_landmarks
        )
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
        # Be sure to not modify input imgs
        imgs = copy.deepcopy(imgs)

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
            if self.use_landmarks:
                boxes, scores, landmarks = self.model(tensors)
            else:
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

            # De-augment results
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "-w",
        "--weights_path",
        type=str,
        help="Path to the networks weights.",
        default="weights/retinaface_mobilenet_0.25.pth",
    )
    parser.add_argument(
        "-noldm", "--no-landmarks", help="Set use of landmarks to false.", action="store_false"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"{args.image_path} do not exist")

    img = cv2.imread(args.image_path)
    if img is None:
        raise ValueError(f"{args.image_path} is invalid")
    input_shape = img.shape

    # Setup model
    face_pixelizer = FacePixelizer(
        input_size=512,
        state_dict_path=args.weights_path,
        device=args.device,
        use_landmarks=args.no_landmarks,
    )

    # Inference
    start = time.time()
    pred = face_pixelizer([img])[0]
    print(f"inference done in {time.time() - start:0.3f} secs.")

    # Plot images
    f, axes = plt.subplots(2, 1)
    for axe, im, title in zip(axes, [img, pred], ["original", "prediction"]):
        axe.imshow(im[..., ::-1])
        axe.set_title(title)
        axe.axis("off")
    plt.tight_layout()
    plt.show()
