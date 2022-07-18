from matplotlib import pyplot as plt
from retinaface.utils import vis_annotations
from retinaface.predict_single import Model
import cv2
import time
import torch
from collections import OrderedDict
import argparse


def get_model(model_path, max_size=512, device="cpu"):
    model = Model(max_size=max_size, device=device)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict["state_dict"].items():
        name = k[6:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image_path", help="Path of the input image.", required=True)
    arg("-o", "--output_path", help="Path of the outputed image.", required=True)
    arg("-m", "--model_path", help="Path of the model.", required=True)
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
    model = get_model(args.model_path, args.max_size, args.device)
    image = cv2.imread(args.image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    start = time.time()
    annotation = model.predict_jsons(image)
    print(f"inference done in {time.time() - start:0.3f} secs")

    plt.imsave(args.output_path, vis_annotations(image, annotation))
