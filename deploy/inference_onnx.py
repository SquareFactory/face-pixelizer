import onnxruntime as ort
import argparse
import time
import numpy as np
from matplotlib import pyplot as plt
import cv2
import albumentations as albu
from retinaface.utils import vis_annotations

ROUNDING_DIGITS = 2


def get_model_onnx(local, model_path, max_size_):
    global max_size
    max_size = max_size_
    if not local:
        return ort.InferenceSession(model_path)
    else:
        raise NotImplementedError

def prepare_image(image: np.ndarray) -> np.ndarray:
    return albu.Compose([albu.LongestMaxSize(max_size=max_size), albu.Normalize(p=1)])(image=image)["image"]




def predict_onnx(ort_session, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = prepare_image(image, max_size)

    # start = time.time()
    outputs = ort_session.run(None, {"input": np.transpose(im, (2, 0, 1))})
    # print(f"inference done in {time.time() - start:0.3f} secs")

    annotations = []

    for box_id, bbox in enumerate(outputs[0]):
        annotations += [
            {
                "bbox": np.round(bbox.astype(float), ROUNDING_DIGITS).tolist(),
                "score": np.round(outputs[1][box_id], ROUNDING_DIGITS),
                "landmarks": np.round(outputs[2][box_id].astype(float), ROUNDING_DIGITS)
                        .reshape(-1, 2)
                        .tolist(),
            }
        ]
    im = albu.Compose([albu.LongestMaxSize(max_size=max_size)])(image=image)["image"]
    return vis_annotations(im, annotations)



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
        type=int
    )
    args = parser.parse_args()
    raw_image = cv2.imread(args.image_path)
    model = get_model_onnx(args.model_path)
    output = predict_onnx(model,raw_image,args.max_size)
    plt.imsave("example.png",output)