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

## Day 5 – Model Evaluation

📌 Overview

On Day 5, the trained Convolutional Neural Network (CNN) model was evaluated using the test dataset. The goal of this step is to measure how well the model generalizes to unseen chest X-ray images.

Evaluation focused on:

model accuracy

model loss

comparison between training and validation performance

identification of misclassifications (optional)


🧠 What was done on Day 5?

✔️ The trained CNN model was tested on the test dataset
✔️ Model performance was measured using loss and accuracy metrics
✔️ Training vs validation curves were plotted
✔️ Optional confusion matrix was generated to analyze predictions

🧪 Evaluation Metrics Used

The following metrics were used to evaluate the model:

Training Accuracy – performance during training

Validation Accuracy – performance on validation data

Test Accuracy – final performance on unseen data

Loss – error value during training and testing


These metrics help determine:

underfitting

overfitting

model generalization ability


📊 Visualization Performed

The following graphs were plotted:

Training vs Validation Accuracy curve

Training vs Validation Loss curve

Optional Confusion Matrix Heatmap


These visualizations help in understanding:

whether the model is improving

whether it is overfitting or underfitting

class-wise prediction performance

🎯 Outcome

The evaluation confirms whether:

the CNN model is reliable

further tuning is needed

architecture needs modification

dataset needs balancing or augmentation

## 📅 Day 6 – Testing & Prediction 🧪


On Day 6, the trained CNN model was tested on unseen chest X-ray images to evaluate its real-world performance.

### ✅ What I did today

✔ Loaded the saved trained model (.h5 file)  
✔ Tested the model with new X-ray images  
✔ Generated predictions (Normal / Pneumonia)  
✔ Evaluated model accuracy on test dataset  
✔ Observed detection performance and reliability  

### 🔍 Output of prediction

- The model takes an input chest X-ray image  
- Processes it using the trained CNN  
- Outputs whether the X-ray is:

👉 *Normal*  
👉 *Pneumonia*

### 🧪 Evaluation Includes

- Test Accuracy  
- Test Loss  
- Correct vs Incorrect Predictions  

### 🚀 Summary of Day 6

The trained model successfully performed:

- Testing on unseen X-ray images  
- Prediction of disease labels  
- Evaluation on the test set  

## 🎉 Day 7 – Final Results & Conclusion 🫁

### 🧠 Project: Chest X-Ray Pneumonia Detection

Day 7 marks the completion of the project. The trained CNN model was tested and the final performance results were obtained.

### ✅ What I completed today

✔ Tested the final model on unseen chest X-ray images  
✔ Displayed final prediction outputs  
✔ Calculated final test accuracy and loss  
✔ Observed confidence scores of predictions  
✔ Prepared final project conclusion  

### 🏁 Final Output

The model successfully predicts:

✔ Normal  
✔ Pneumonia  

For each image, the model also provides:

✔ Predicted class  
✔ Confidence percentage  

### 📊 Final Evaluation Summary

- Model Accuracy: ✔️  
- Model Loss: ✔️  
- Reliable pneumonia detection results  

### 🧠 Learning Outcomes

✔ Learned dataset preprocessing  
✔ Built CNN model  
✔ Trained model  
✔ Evaluated performance  
✔ Made predictions  
✔ Interpreted final results  

### 🚀 Final Conclusion

The project successfully demonstrates:

- Deep learning for medical imaging  
- Automatic pneumonia detection  
- Chest X-ray image classification
