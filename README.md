# AreUnemia-Models

Welcome to the AreUnemia application repository. This repository contains the machine learning models and data engineering processes utilized to develop and deploy the AreUnemia application, aimed at early detection of anemia using smartphone images of the eye conjunctiva. The repository is divided into two main sections: Conjunctiva Segmentation and Anemia Prediction. 

## Table of Contents

1. Data Engineering
2. Models
3. Results
4. Acknowledgements

## Data Engineering

Data engineering is a critical step in developing robust machine learning models. For this project, we utilized two distinct datasets:

1. Conjunctiva Segmentation Data:
    - Source: RoboFlow
    - Data Description: 333 annotated images used to train the model to identify and segment the conjunctiva region of the eye.

2. Anemia Prediction Data:
    - Source: Mendeley
    - Data Description: 8524 images labeled as either 'anemia' or 'not-anemia'. This dataset was used to train the classification model to predict anemia.

## Models
### Conjunctiva Segmentation
**Objective**: To predict the location of the eye conjunctiva, crop it, and output the segmented image.

- Model Architecture: YOLOv8 (You Only Look Once version 8)
- Dataset: 333 images from RoboFlow
- Output: Segmented images highlighting the conjunctiva region
- File Format: PyTorch model (.pt)

**Steps**:
- Data Preprocessing: Images were annotated and preprocessed by source (RoboFlow).
- Model Training: The YOLOv8 architecture was employed for training the segmentation model.
- Inference: The trained model predicts the location of the conjunctiva, crops it from the original image, and outputs the cropped image.

### Anemia Prediction
**Objective**: To predict whether the user has anemia based on the segmented conjunctiva images.

- Model Architecture: Custom Convolutional Neural Network (CNN)
- Dataset: 8524 images from Mendeley, labeled for binary classification (0 - anemia, 1 - not-anemia)
- Performance:
    - Training Accuracy: 0.99
    - Validation Accuracy: 0.98
- File Format: Keras model (.h5)

**Steps**:
- Data Preprocessing: Images were preprocessed and normalized.
- Model Training: A CNN was trained to classify the images into 'anemia' or 'not-anemia'.
- Inference: The model takes the cropped images from the Conjunctiva Segmentation model and predicts the likelihood of anemia.

## Results
- Conjunctiva Segmentation Model: The YOLOv8-based segmentation model accurately identifies and crops the conjunctiva region from input images.
- Anemia Prediction Model: The CNN classification model achieves a high accuracy of 0.99 on the training set and 0.98 on the validation set, providing reliable predictions on anemia status.

## Acknowledgements
We would like to thank the contributors of the datasets used in this project:
- RoboFlow for providing the conjunctiva segmentation dataset.
- Mendeley for the anemia classification dataset.

For more details, please refer to the documentation provided in the repository or contact us.

