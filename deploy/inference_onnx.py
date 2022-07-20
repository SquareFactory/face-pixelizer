import onnxruntime as ort
import argparse
import torch
import time
import numpy as np
import cv2
import albumentations as albu
from retinaface.utils import vis_annotations

def prepare_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    image = albu.Compose([albu.LongestMaxSize(max_size=max_size), albu.Normalize(p=1)])(image=image)["image"]

    height, width = image.shape[:2]

    return cv2.copyMakeBorder(image, 0, max_size - height, 0, max_size - width, borderType=cv2.BORDER_CONSTANT)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image_path", help="Path of the input image.", required=True)
    # arg("-o", "--output_path", help="Path of the outputed image.", required=True)
    arg("-m", "--model_path", help="Path of the onnx model.")
    arg(
        "-ms",
        "--max_size",
        help="The max size of the image to resize.",
        required=False,
        default=256,
    )
    args = parser.parse_args()
    image = cv2.imread(args.image_path)
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = prepare_image(im, args.max_size)

    ort_session = ort.InferenceSession(args.model_path)
    start = time.time()
    # outputs = ort_session.run(None, {"input": np.expand_dims(np.transpose(image, (2, 0, 1)), 0)})
    print(np.expand_dims(im.astype(np.float32),0).shape)
    outputs = ort_session.run(None, {"input": np.expand_dims(np.transpose(im, (2, 0, 1)).astype(np.float32),0)})
    print(f"inference done in {time.time() - start:0.3f} secs")

    annotations = []

    for box_id, box in enumerate(outputs[0]):
        annotations += [
            {
                "bbox": box.tolist(),
                "score": outputs[1][box_id],
                "landmarks": outputs[2][box_id].reshape(-1, 2).tolist(),
            }
        ]

    image = albu.Compose([albu.LongestMaxSize(max_size=args.max_size)])(image=image)["image"]
    cv2.imwrite("example.jpg", vis_annotations(image, annotations))