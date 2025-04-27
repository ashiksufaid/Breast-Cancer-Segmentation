# Breast cancer segmentation from ultrasound images

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [File Descriptions](#file-descriptions)

## Introduction
This project presents a deep learning model that segments breast cancer regions in ultrasound images. By exploiting the characteristic appearance of cancerous tissue, the algorithm generates a binary mask that highlights cancerous areas. 
It takes breast ultrasound images in png format as the input and outputs a boolean mask with pixel value 1 for cancerous regions and 0 for others. If the Ultrasound
image of a normal breast is given as input, it should output a mask with all pixel values as zero.

## Dataset
[Kaggle Breast Ultrasound Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) was used to train this model. It contains ultrasound images and corresponding masks for normal, benign, and malignant breast cases. All three sets of images were used to train this model.

## Model Architecture
U-Net.
Its architecture consists of two main parts: an encoder and a decoder. The encoder gradually
reduces the spatial dimensions of the input image. The decoder then upsamples these features
back to the original resolution, which helps reconstruct the spatial details. Another component of
the U-Net is skip connections which connect the corresponding layers of encoder and decoder.
This ensures that the spatial features are not lost due to down sampling by the encoder.

## File Descriptions
### data
This folder contains three subfolders: normal, benign and malignant. Each folder contains ultrasound images belonging to each class and their corresponding masks. **This folder also contains 10 ultrasound images for predict.py to make predictions**
### config.py
This file contains importand variables such as 
- batchsize
- epochs
- device
- optimizer
- loss
### dataset.py
The dataset.py file contains two classes - BreastCancerDataset and CustomDataLoader.
- The `BreastCancerDataset` takes in the path to the folder with the images as the argument. The folder should contain three subfolders: normal, benign and malignant. These subfolders should contain the images and corresponding masks. The format is similar to what is found in the `data` folder.
- The `CustomDataLoader` takes in the dataset, batchsize and shuffle(boolean) as the input. It handles batching and optional shuffling of the dataset. It manually iterates through the dataset, fetching batches of images and corresponding masks, and stacks them into tensors for training and evaluation.   
### model.py
This file defines the U-Net architecture used for breast cancer segmentation. The model follows an encoder–decoder structure:

- The **encoder** path (`Down` blocks) progressively downsamples the input while capturing spatial features.
- The **decoder** path (`Up` blocks) upsamples the feature maps and uses skip connections to recover fine-grained details.
- Each block uses two convolutional layers followed by batch normalization and ReLU activation (`DoubleConv`).
- The final output layer (`OutConv`) uses a 1×1 convolution to generate a single-channel segmentation mask.

The model is implemented in PyTorch and can handle input images with 3 channels and outputs a binary mask highlighting cancerous regions.
### train.py
This file defines the training loop for the segmentation model. It moves the model to the specified device, iteratively optimizes the network using the given loss function and optimizer, and tracks the training loss across epochs. 
### predict.py
This file contains the function `plot_segmented_images` which takes the directory which the images are stored as the input and outputs the segmented masks by the model alongside the given images. By default, it takes the `data` folder in this repository as the input. The file fetch model from `model.py` and weights from `checkpoints` folder for evaluation. **No matter the number of image given as input, the function will only provide predictions for the first 5 images for the sake of interpretability**. 
**The function is equiped to avoid files with `_mask` in it avoid making predictions for mask images. So you can input image paths in a whole folder without separating mask files**
## References

Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.
