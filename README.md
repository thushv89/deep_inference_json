# Inferencing a Pre-trained VGG-16 Model for Custom Images

## Technical tools
Tensorflow: 1.0.0
Python: 3.4

## Introduction
This repository implements a script for inferencing the classes of custom images using a pretrained deep network (VGG-16). The weights (parameters) are found at (https://www.cs.toronto.edu/~frossard/post/vgg16/)
The architecture of the model can be found in this paper (https://arxiv.org/pdf/1409.1556.pdf)

## How to run
Run the script vgg_inference.py
The script might download the weights (~ 500MB) and will require few minutes
Once weights are downloaded tensorflow will load those parameters in to tensorflow variables
