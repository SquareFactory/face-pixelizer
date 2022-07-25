from matplotlib import pyplot as plt
from retinaface.utils import vis_annotations
from retinaface.predict_single import Model
import cv2
import time
import torch
from collections import OrderedDict
import argparse
from retinaface.pre_trained_models import get_model as get_model_url
import onnxruntime as ort
import numpy as np
import albumentations as albu
import pathlib

ROUNDING_DIGITS = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(model_path = None, max_size=512):
    if model_path is not None:
        if pathlib.Path(model_path).suffix == "onnx":
            return get_model_onnx(model_path, max_size)
        else:
            model = Model(max_size=max_size, device=device)
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = OrderedDict()
            for k, v in state_dict["state_dict"].items():
                name = k[6:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            return model
        
    else: return get_model_url("resnet50_2020-07-20", max_size, device)
        


def predict(model, image):
    with torch.no_grad():
        annotation = model.predict_jsons(image)
    if len(annotation) == 1 and annotation[0]["score"]==-1: return image
    return vis_annotations(image, annotation)


def get_model_onnx(model_path, max_size_):
    global max_size
    max_size = max_size_
    return ort.InferenceSession(model_path)

def prepare_image(image: np.ndarray) -> np.ndarray:
    return albu.Compose([albu.LongestMaxSize(max_size=max_size), albu.Normalize(p=1)])(image=image)["image"]

def predict_onnx(ort_session, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = prepare_image(image)
    outputs = ort_session.run(None, {"input": np.transpose(im, (2, 0, 1))})
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
    if len(annotations) == 1 and annotations[0]["score"]==-1: return im
    return vis_annotations(im, annotations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image_path", help="Path of the input image.", required=True)
    arg("-l", "--local_model", help="if the model is local", action='store_true')
    arg("-o", "--output_path", help="Path of the outputed image.", required=True)
    arg("-m", "--model_path", help="Path of the model.")
    arg(
        "-ms",
        "--max_size",
        help="The max size of the image to resize.",
        required=False,
        default=512,
    )
    arg(
        "-d",
        "--device",
        help="The device to infer on (either cpu or cuda).",
        required=False,
        default="cpu",
    )
    args = parser.parse_args()
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model = get_model(args.local_model, args.model_path, args.max_size)
    model.eval()
    start = time.time()
    output = predict(model,image)
    print(f"inference done in {time.time() - start:0.3f} secs")
    plt.imsave(args.output_path,output)
