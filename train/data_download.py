"""Copyright (C) SquareFactory SA - All Rights Reserved.
This source code is protected under international copyright law. All rights 
reserved and protected by the copyright holders.
This file is confidential and only available to authorized individuals with the
permission of the copyright holders. If you encounter this file and do not have
permission, please contact the copyright holders and delete this file.
"""
import argparse
import shutil
from pathlib import Path

import wget


def download_data(
    train_set: bool,
    valid_set: bool,
    labels: bool,
    unzip: bool,
):
    # create a data dir if not present
    path_to_data = Path("data/")
    path_to_data.mkdir(parents=True, exist_ok=True)

    # download training set
    if train_set:
        url = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_train.zip"
        outp = path_to_data / "WIDER_train.zip"
        wget.download(url, str(outp))
        if unzip:
            shutil.unpack_archive(outp, path_to_data / "WIDER_train")

    # download validation set
    if valid_set:
        url = (
            "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip"
        )
        outp = Path(path_to_data / "WIDER_val.zip")
        wget.download(url, str(outp))
        if unzip:
            shutil.unpack_archive(outp, path_to_data / "WIDER_val")

    # download labels
    if labels:
        url = "https://drive.google.com/uc?id=1oI4I313hOrmO9G312KuYC-kx35fy922T"
        outp = path_to_data / "wider_annotations.zip"
        wget.download(url, str(outp))
        if unzip:
            shutil.unpack_archive(outp, path_to_data / "WIDER_labels")


def txt_to_json(path_to_txt: str) -> None:
    with open(path_to_txt, "r") as f:
        ds = f.read()
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ts",
        "--train-set",
        action="store_true",
        help="Download the training set",
        default=False,
    )
    parser.add_argument(
        "-vs",
        "--val-set",
        action="store_true",
        help="Download the validation set",
        default=False,
    )
    parser.add_argument(
        "-lbl",
        "--labels",
        action="store_true",
        help="Download the annotations",
        default=False,
    )
    parser.add_argument(
        "-a", "--all", action="store_true", help="Download all", default=False
    )
    parser.add_argument(
        "-z",
        "--unzip",
        action="store_true",
        help="unzip all the downloaded files",
        default=False,
    )
    args = parser.parse_args()
    if args.all:
        download_data(True, True, True, args.unzip)
    else:
        download_data(args.train_set, args.val_set, args.labels, args.unzip)
