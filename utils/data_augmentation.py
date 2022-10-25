import random
from typing import Tuple

import numpy as np
from box_utils import matrix_iof


def random_crop(
    image: np.ndarray, boxes: np.ndarray, labels: np.ndarray, img_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Crop random patch.
    if random.uniform(0, 1) <= 0.2:
        scale = 1.0
    else:
        scale = random.uniform(0.3, 1.0)
    """
    height, width = image.shape[:2]
    pad_image_flag = True

    for _ in range(250):

        pre_scales = [0.3, 0.45, 0.6, 0.8, 1.0]
        scale = random.choice(pre_scales)  # nosec
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        if width == w:
            unclear_variable = 0
        else:
            unclear_variable = random.randrange(width - w)  # nosec
        if height == h:
            t = 0
        else:
            t = random.randrange(height - h)  # nosec
        roi = np.array((unclear_variable, t, unclear_variable + w, t + h))

        value = matrix_iof(boxes, roi[np.newaxis])
        flag = value >= 1
        if not flag.any():
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        labels_t = labels[mask_a].copy()

        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1] : roi[3], roi[0] : roi[2]]

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] = boxes_t[:, :2] - roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        boxes_t[:, 2:] = boxes_t[:, 2:] - roi[:2]

        # make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, pad_image_flag
    return image, boxes, labels, pad_image_flag


def random_horizontal_flip(
    image: np.ndarray,
    boxes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    width = image.shape[1]
    if random.randrange(2):  # nosec
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]

    return image, boxes


def _pad_to_square(image: np.ndarray, pad_image_flag: bool) -> np.ndarray:
    if not pad_image_flag:
        return image
    height, width = image.shape[:2]
    long_side = max(width, height)
    image_t = np.zeros((long_side, long_side, 3), dtype=image.dtype)
    image_t[:height, :width] = image
    return image_t


class Preproc:
    def __init__(self, img_dim: int) -> None:
        self.img_dim = img_dim

    def __call__(
        self, image: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if targets.shape[0] == 0:
            raise ValueError("this image does not have gt")

        boxes = targets[:, :4].copy()
        labels = targets[:, -1:].copy()

        image_t, boxes_t, labels_t, pad_image_flag = random_crop(
            image, boxes, labels, self.img_dim
        )

        image_t = _pad_to_square(image_t, pad_image_flag)
        image_t, boxes_t = random_horizontal_flip(image_t, boxes_t)
        height, width = image_t.shape[:2]

        boxes_t[:, 0::2] = boxes_t[:, 0::2] / width
        boxes_t[:, 1::2] = boxes_t[:, 1::2] / height

        targets_t = np.hstack((boxes_t, labels_t))

        return image_t, targets_t
