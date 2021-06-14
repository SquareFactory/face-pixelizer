# isquare.ai - python example
*In this tutorial, we will cover all steps to deploy a model written in python to isquare.ai. This tutorial’s duration is about 30mins, and is addressed at ML developers, software engineers or project managers who want to put their machine learning models into production in a few clicks.*

## Use case: Face Pixelizer
With the rapid rise of authoritarian laws like “sćurité globale” in France or “MPT” in switzerland, it could become illegal to publish pictures or videos showing faces of policeman. Since these laws come with a rise in police violence, it is super important to continue filming the interventions of law officiers. Therefore, we developed a machine learning model which automatically blurs faces in any image, and is capable of doing so in real time. In this tutorial, we will show how to deploy this model efficiently with isquare.ai.
![example](imgs/plot.jpg)

## Step 1: Make your code isquare compatible
*Before deploying our model, we have to make sure that it is compatible with the platform. From an existing model this is easily achieved in a few simple steps.*

### Step 1.1: Set up your environment
Most deep learning models are not coded from scratch and depend on external libraries (e.g. python, tensorflow). With isquare.ai, all requirements are handled by a Dockerfile, which is basically a set of instructions which sets up an environment. If you’re new to Docker, check the [documentation](https://docs.docker.com/engine/reference/builder/). Our face pixelizer was build with pytorch and has following python dependencies:
```
albumentations==0.5.1
matplotlib==3.4.1
numpy==1.19.2
torch==1.7.0
torchvision==0.8.1
``` 
These are saved to a file at the root of the directory, called requirements.txt. Other than the dependencies, we also have the weights (retinaface_moblenet_0.25.pth) of the trained model which are located at the root of our repository, along with the requirements file.

## Step 2: Deploy your model
