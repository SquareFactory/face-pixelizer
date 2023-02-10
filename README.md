![iSquare](imgs/iSquare-color.png)
# Face-pixelizer
Welcome to Squarefactory's official face pixeliser repository. In this repository, you will find all code necessary to train and deploy a custom retinaface model using isquare.ai.
You can follow the [tutorial](docs.isquare.ai) as well as the material in this repository to learn how to train and deploy your machine learning models using isquare.ai. At the end of the tutorial, you'll know how to:
- Train a model using isquare.ai (You will not have to change your code)
- Write a deployment script as well as an environment file for you trained model
- Deploy the model and learn how to send a video stream (your webcam) and perform real-time inference with a deep learning model


## Start locally
Before using an MLOPS platform to monitor and scale your trainings and deployments, you can test your models locally. Since you may not have sufficient coomputing power to train your model using your machine, we will only test our deployment script locally. To do this, you need to first configure your environment and install the relevant python packages.

```bash
conda create -n face-pixelizer python=3.8
conda activate face-pixelizer
pip install -r requirements.txt
```
After this, you're all set to test your model locally. If you have a cuda-compatible GPU, it will be used, but the script will also run in reasonable time on your CPU. We provide an picture taken at the street parade in Zurich as example image, but you can use any private image with this script. However, to see an effect, the picture should contain at least one face.
Let's try it out


```bash
python face_pixelizer.py --image_path imgs/example_01.jpg
```
If you used our provided image, you should see the following output:

![example](imgs/plot.jpg)

## Scale your training using iSquare
We kindly recomend following the [tutorial](docs.isquare.ai). You can use iSquare to scale your training to better hardware, perform parallel hyperparameter sweeps, monitor and compare your models, and much more. 
Once you've deployed your model, you can interact with it using our [python client](https://github.com/SquareFactory/i2-cli). You can perform single inference, stream data asynchronously and integrate deep learning models in any of your code by adding just 3 lines.