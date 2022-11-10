from copy import deepcopy
from itertools import product
from math import ceil

import cv2
import numpy as np
import torch


def get_prior_box(
    height,
    width,
    min_sizes=[[16, 32], [64, 128], [256, 512]],
    steps=[8, 16, 32],
    clip=False,
):
    """Compute prior box.
    <TODO> vectorize to increase speed
    """

    feature_maps = [[ceil(height / step), ceil(width / step)] for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes[k]:
                s_kx = min_size / width
                s_ky = min_size / height
                dense_cx = [x * steps[k] / width for x in [j + 0.5]]
                dense_cy = [y * steps[k] / height for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]
    output = torch.Tensor(anchors).view(-1, 4)

    if clip:
        output.clamp_(max=1, min=0)

    return output


def decode_boxes(boxes, priors, variances):
    """Decode locations from predictions using priors.
    To undo the encoding we did for offset regression at train time.
    Args:
        boxes (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat(
        (
            priors[:, :2] + boxes[:, :, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(boxes[:, :, 2:] * variances[1]),
        ),
        dim=-1,
    )
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


# def decode_landmarks(landmarks, priors, variances):
# """Decode landm from predictions using priors.

# To undo the encoding we did for offset regression at train time.

# Args:
# landmarks (tensor): landm predictions for boxes layers,
# Shape: [num_priors,10]
# priors (tensor): Prior boxes in center-offset form.
# Shape: [num_priors,4].
# variances: (list[float]) Variances of priorboxes
# Return:
# decoded landm predictions
# """
# return torch.cat(
# (
# priors[:, :2] + landmarks[:, :, :2] * variances[0] * priors[:, 2:],
# priors[:, :2] + landmarks[:, :, 2:4] * variances[0] * priors[:, 2:],
# priors[:, :2] + landmarks[:, :, 4:6] * variances[0] * priors[:, 2:],
# priors[:, :2] + landmarks[:, :, 6:8] * variances[0] * priors[:, 2:],
# priors[:, :2] + landmarks[:, :, 8:10] * variances[0] * priors[:, 2:],
# ),
# dim=-1,
# )


def pixelize(img, blocks=6):
    """Pixelize an imagee.
    taken from:
        pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python
    Args:
        img (np.array): image to pixellize
        blocks: number of different blocks
    """
    h, w = img.shape[:2]
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")
    for y in range(1, len(y_steps)):
        for x in range(1, len(x_steps)):
            # compute the starting and ending (x, y)-coordinates
            # for the current block
            x_start = x_steps[x - 1]
            y_start = y_steps[y - 1]
            x_end = x_steps[x]
            y_end = y_steps[y]

            # extract the ROI using NumPy array slicing, compute the
            # mean of the ROI, and then draw a rectangle with the
            # mean RGB values over the ROI in the original img
            roi = img[y_start:y_end, x_start:x_end]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (B, G, R), -1)
    return img


def clip_all_boxes(boxes, img_dim, in_place=False):
    if not in_place:
        boxes = deepcopy(boxes).cpu().numpy()

    h, w = img_dim
    boxes[np.where(boxes[:, 0] < 0), 0] = 0
    boxes[np.where(boxes[:, 1] < 0), 1] = 0
    boxes[np.where(boxes[:, 2] > h - 1), 2] = h - 1
    boxes[np.where(boxes[:, 3] > w - 1), 3] = w - 1
    x_max_smallerthan_min = np.where(boxes[:, 0] > boxes[:, 2] - 1)
    y_max_smallerthan_min = np.where(boxes[:, 1] > boxes[:, 3] - 1)
    boxes[x_max_smallerthan_min, 0] = boxes[x_max_smallerthan_min, 2] - 1
    boxes[y_max_smallerthan_min, 1] = boxes[y_max_smallerthan_min, 3] - 1
    return torch.tensor(boxes)
