# AI-based-Lung-abnormalities-detection-from-chest-X-rays
7 days mini project using python,AIML

## DAY-1 PROBLEM STUDY

## PROJECT ABSTRACT

Early identification of lung abnormalities is essential for effective treatment and disease management.This project presents an AI-based system that analyzes chest X-ray images using deep learning tecniques to detect abnormal network is trained to extract visual features and classify X-rays as normal or abnormal.The system supports early screening and assists healthcare professionals in clinical decision-making.



## 🔎OBJECTIVE FOR DAY-1

The goal of Day 1 is to collect the required dataset and set up the development environment for the project.Today's focus is only on preparing the resources needed for model building in upcoming days.


## DATASET LINK  https://drive.google.com/file/d/1bj8DGr3GcXncarTgoMxJtR4hP57XY47A/view?usp=sharing


## DAY 2- DATA PREPROCESSING

** objective **
  Prepare the chest X-ray dataset for model training by:

> Organizing images into train/test/validation sets
> resizing images to a fized size
> removing unreadable or corrupted images
> ensuring class folders are correct

 ## PREPROCESSING STEPS PERFORMED 

 % Loaded dataset from source folder
 % Verified paths and counted images
 % split dataset into 
   * 70% train
   * 20% test
   * 10% validation
% Resized all images to 224x224 pixels
% converted images to grayscale/RGB as required
% removed :
   * duplicated images
   * unreadable files
   * zero-byte corrupted images


## DAY 3 - CNN model design 

OVERVIEW :

Today focus on designing a CNN for chest X-ray images.The model architecture is built to extract features from x-rays images through multiple convolution and pooling layer, followed by fully connected layer.

MODEL ARCHITECTURE 

1. Input layer
   - Accepts imagesof size 224*224*3
2.convloution layer 1
   - conv2D:32 filters,3x3 kernel,ReLU activation
   - Maxpooling2D: 2x2 pool size
3. Convolutional layer 2
   - conv2D :64 filters,3x3 kernel,ReLU activation
   - Maxpooling2D: 2x2 pool size
4.  Convolutional layer 3
   - conv2D :128 filters,3x3 kernel,ReLU activation
   - Maxpooling2D: 2x2 pool size
5. Flattern layer
   - converts 2D feature maps into a 1D vector
6.Fully connected layer
   - dropout:0.5
7.output layer
   - dense layer with neurons depending on the number of classes
   - softmax activation

## Day 4 – Model Training

📌 Overview

On Day 4, the focus was on model selection, design, and training for the Chest X-ray image classification project. A Convolutional Neural Network (CNN) was chosen because CNNs are highly effective for extracting important spatial features from medical images such as chest X-rays.

The model was trained to classify X-ray images into multiple categories such as:

Normal
Pneumonia

🧠 Why CNN for Chest X-ray?

CNNs are ideal for this task because they can automatically learn:

edges and contours of lungs

textures and opacities

disease-related patterns

structural abnormalities

🏗️ Model Design & Architecture

The model is optimized using:

Adam optimizer

Categorical cross-entropy loss

Accuracy as evaluation metric

🏋️ Training Strategy

The dataset was divided into:

Training set

Validation set

Test set

Data augmentation was applied to improve generalization:

rotation

zoom

horizontal flip

rescaling


The model was trained for multiple epochs while monitoring:

training accuracy

validation accuracy

training loss

validation loss

📊 Performance Evaluation

After training, the model was evaluated using:

Test accuracy

Accuracy curve

Loss curve

