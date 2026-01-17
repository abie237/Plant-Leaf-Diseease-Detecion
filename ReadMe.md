#  Image Classification using InceptionV3

This project is an image classification deep learning model built using InceptionV3 in Keras (TensorFlow backend).  
The model classifies wire images into three categories:

- Red Hot  
- Healthy  
- Red Rust  

The purpose of this project is to automatically detect the condition of wires using image data and deep learning.

---

##  Project Overview

Manual inspection of wire conditions can be slow and unreliable. This project uses transfer learning with the InceptionV3 convolutional neural network to classify wire condition images accurately.

The model is trained on a custom dataset and fine-tuned for better performance on the given classes.

---

##  Model Architecture

- Base Model: InceptionV3 (pre-trained on ImageNet)
- Framework: Keras (TensorFlow)
- Programming Language: Python
- Technique: Transfer Learning
- Input Image Size: 299 x 299 (RGB)
- Number of Classes: 3

---

## 📂 Dataset Structure

The dataset is organized into three classes:


dataset/
├── Red_Hot/
├── Healthy/
└── Red_Rust/




Each folder contains images related to that class.

---

## Requirements

- Python 3.8 or higher  
- TensorFlow  
- Keras   
- OpenCV (optional)  

Install dependencies using: 




---
##  Training the Model

Steps followed to train the model:

1. Prepare and organize the dataset into training and validation folders
2. Load the InceptionV3 model with ImageNet weights
3. Freeze base model layers
4. Add custom fully connected layers
5. Compile and train the model
6. Save the trained model

---

##  Model Testing and Prediction

To predict the class of a new image:

1. Load the trained model
2. Preprocess the image to match InceptionV3 input requirements
3. Perform prediction
4. Map the output to one of the class labels:
   - Red Hot
   - Healthy
   - Red Rust

---

##  Results

- The model is able to classify wire images into three categories
- Transfer learning improves accuracy with limited data
- Performance can be improved with more training data

---

##  Future Improvements

- Permit the model to detect if an object if a leaf or not

---

##  Contributing

Contributions are welcome.  
Feel free to fork the repository and submit pull requests.

---

---



