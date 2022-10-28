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
    if not path_to_data.exists():
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
        url = "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"
        outp = path_to_data / "wider_face_split.zip"
        wget.download(url, str(outp))
        if unzip:
            shutil.unpack_archive(outp, path_to_data / "WIDER_labels")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ts", "--train_set", type=bool, help="Download the train_set", default=True
    )
    parser.add_argument(
        "-vs", "--val_set", type=bool, help="Download the val_set", default=True
    )
    parser.add_argument(
        "-lbl", "--labels", type=bool, help="Download the annotations", default=True
    )
    parser.add_argument(
        "-z", "--unzip", type=bool, help="unzip all the downloaded files", default=True
    )
    args = parser.parse_args()
    download_data(args.train_set, args.val_set, args.labels, args.unzip)
