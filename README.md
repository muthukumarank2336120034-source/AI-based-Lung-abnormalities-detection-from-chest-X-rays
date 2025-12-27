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
