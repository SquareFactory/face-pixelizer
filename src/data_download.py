import gdown
import shutil
import json
import numpy as np
import argparse


def donwload_data(
    train_set: bool,
    valid_set: bool,
    unzip: bool,
):
    # download annotations
    # if annotations:
    #     url = "https://drive.google.com/uc?id=1QO177uqTpIhjauy3WUIMjD4T5u1SzGcS"
    #     output = "annotations.zip"
    #     gdown.download(url, output, quiet=False)
    #     if unzip:
    #         shutil.unpack_archive(output, "annotations")

    # download training set
    if train_set:
        url = "https://drive.google.com/uc?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M"
        output = "WIDER_train.zip"
        gdown.download(url, output, quiet=False)
        if unzip:
            shutil.unpack_archive(output, "WIDER_train")

    # download validation set
    if valid_set:
        url = "https://drive.google.com/uc?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q"
        output = "WIDER_val.zip"
        gdown.download(url, output, quiet=False)
        if unzip:
            shutil.unpack_archive(output, "WIDER_val")


# def convert_annotations(input_path, output_path):
#     result = []
#     temp = {}
#     valid_annotation_indices = np.array([0, 1, 3, 4, 6, 7, 9, 10, 12, 13])
#     with open(input_path) as f:
#         for line_id, line in enumerate(f.readlines()):
#             if line[0] == "#":
#                 if line_id != 0:
#                     result += [temp]
#                 temp = {"file_name": line.replace("#", "").strip(), "annotations": []}
#             else:
#                 points = line.strip().split()

#                 x_min = int(points[0])
#                 y_min = int(points[1])
#                 x_max = int(points[2]) + x_min
#                 y_max = int(points[3]) + y_min

#                 x_min = max(x_min, 0)
#                 y_min = max(y_min, 0)

#                 x_max = max(x_min + 1, x_max)
#                 y_max = max(y_min + 1, y_max)

#                 landmarks = np.array([float(x) for x in points[4:]])

#                 if landmarks.size > 0:
#                     landmarks = (
#                         landmarks[valid_annotation_indices].reshape(-1, 2).tolist()
#                     )
#                 else:
#                     landmarks = []

#                 temp["annotations"] += [
#                     {"bbox": [x_min, y_min, x_max, y_max], "landmarks": landmarks}
#                 ]

#         result += [temp]

#     with open(output_path, "w") as f:
#         json.dump(result, f, indent=2)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-ts", "--train_set", type=bool, help="Download the train_set", default=True
)
parser.add_argument(
    "-vs", "--val_set", type=bool, help="Download the val_set", default=True
)
# parser.add_argument(
#     "-a", "--annotations", type=bool, help="Download the annotations", default=True
# )
parser.add_argument(
    "-z", "--unzip", type=bool, help="unzip all the downloaded files", default=True
)
args = parser.parse_args()
donwload_data(args.train_set, args.val_set, args.unzip)
# convert_annotations("annotations/train/label.txt", "annotations/train/label.json")
# convert_annotations("annotations/val/label.txt", "annotations/val/label.json")