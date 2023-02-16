![iSquare](imgs/iSquare-color.png)

# Face-pixelizer
Welcome to Squarefactory's official face pixeliser repository. In this repository, you will find all code necessary to train and deploy a custom retinaface model using Squarefactory's MLOPS platform, [Isquare](https://app.isquare.ai). \

This repository is an implementation of the retinaface model from [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/pdf/1905.00641.pdf). \

You can follow the [tutorial](docs.isquare.ai) as well as the material in this repository to learn how to train and deploy your machine learning models using isquare.ai. At the end of the tutorial, you'll know how to:
- Train a model using isquare.ai (You will not have to change your code) \
- Write a deployment script as well as an environment file for you trained model \
- Deploy the model and learn how to send a video stream (your webcam) and perform real-time inference with a deep learning model. \

The following set-up instruction are for use on a local machine.

## Set-up (local)
The versions of torch, torchtext and torchvision which are in the [requirements](train/train_requirements.txt) correspond to the ones that come pre-installed on the nvidia-image that we use for training on isquare. As these version do not exist in pip, a slight edit of the requirements is necessary before performing a local set up. \
Simply comment out the first 3 requirements and  un-comment the commented ones, as explained on the requirement file. Then, the package can be installed as follows : 

```
conda create -n face-pixelizer python=3.8
conda activate face-pixelizer
pip install retinaface_pytorch==0.0.8 --no-deps
pip install -e .
pip install opencv-python==4.5.3.36
```
retinaface-pytorch needs to be installed separately as its requirements are out-dated, but this does not cause any compatibility issue.

## Download the data
The WIDERFACE dataset can be downloaded using this [script](train/data_download.py). The wget package is needed for this.
```
pip install wget==3.2
python train/data_download.py -az
```

## Launch a training
```
python train/train.py --epochs <N_EPOCHS> 
```
additionnal arguments : \
--no-landmarks : trains a model to perform facial detection alone, without facial landmarks detection. Probably impairs performance. \
--weights_path : path to pretrained weights if you want to start with a pretrained model. (The backbone weights are pretrained by default. Path to these weights can be changed in the [config](retina/config.yml).) \
--config : change path to config file. (default : ./retina/config.yml) \

After this, you're all set to test your model locally. If you have a cuda-compatible GPU, it will be used, but the script will also run in reasonable time on your CPU. We provide an picture taken at the street parade in Zurich as example image, but you can use any private image with this script. However, to see an effect, the picture should contain at least one face.
Let's try it out

## Example

```
python face_pixelizer.py --image_path imgs/example_01.jpg
```

![example](imgs/plot.jpg)

## Aknowledgements
Some of the code and the utilities used in this repository are taken from the [retinaface-pytorch](https://github.com/ternaus/retinaface) package, sometimes adapted. \
