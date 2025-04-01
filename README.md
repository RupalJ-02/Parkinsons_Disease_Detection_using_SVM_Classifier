# Parkinson's Disease Detection and Prediction Using Deep Learning

## Overview

This project aims to develop a machine learning model for predicting Parkinson's disease using voice measurements. Parkinson's disease is a progressive brain disorder that causes movement problems, mental health issues, and other health problems. It's a neurodegenerative disease that affects the central nervous system.

## Objective

Early detection of Parkinson's disease is vital for effective treatment and management. This project uses a dataset of biomedical voice measurements to train an SVM model to predict the presence of Parkinson's disease. The model can assist medical professionals in diagnosing Parkinson's disease and improving patient care.

## Importance

Early detection of Parkinson's disease is vital for effective treatment and management, which can significantly improve the quality of life for patients.

## Approach

1. **Data Collection:** A dataset of biomedical voice measurements is used.
2. **Data Preprocessing:** The dataset is cleaned and preprocessed to handle missing values and outliers.
3. **Model Training:** An SVM model is trained using the preprocessed data.
4. **Model Evaluation:** The performance of the trained model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## DATASET
## Attribute Information:
The dataset contains the following attributes:
Title: Parkinsons Disease Data Set
Abstract: Oxford Parkinson's Disease Detection Dataset

* **name**: ASCII subject name and recording number.
* **MDVP:Fo(Hz)**: Average vocal fundamental frequency.
* **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency.
* **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency.
* **MDVP:Jitter(%)**, **MDVP:Jitter(Abs)**, **MDVP:RAP**, **MDVP:PPQ**, **Jitter:DDP**: Several measures of variation in fundamental frequency.
* **MDVP:Shimmer**, **MDVP:Shimmer(dB)**, **Shimmer:APQ3**, **Shimmer:APQ5**, **MDVP:APQ**, **Shimmer:DDA**: Several measures of variation in amplitude.
* **NHR**, **HNR**: Two measures of the ratio of noise to tonal components in the voice.
* **status**: Health status of the subject (1 = Parkinson's, 0 = healthy).
* **RPDE**, **D2**: Two nonlinear dynamical complexity measures.
* **DFA**: Signal fractal scaling exponent.
* **spread1**, **spread2**, **PPE**: Three nonlinear measures of fundamental frequency variation.

## Impact

The model can assist medical professionals in diagnosing Parkinson's disease and improving patient care. Early detection and prediction of Parkinson's disease can lead to timely interventions and better management of the disease.


## Dataset

The dataset used in this project is a publicly available dataset of biomedical voice measurements. It contains features extracted from voice recordings of individuals with and without Parkinson's disease.

## Model

The machine learning model used in this project is a Support Vector Machine (SVM). An SVM is a supervised learning algorithm that can be used for both classification and regression tasks. In this project, the SVM is used to classify individuals as having or not having Parkinson's disease based on their voice measurements.

## Results

## Model Evaluation Metrics

Here are the evaluation metrics for the model:

* **Accuracy: 0.8846153846153846**
    * Accuracy represents the overall correctness of the model's predictions. It's the ratio of correctly predicted instances to the total instances. In this case, the model correctly predicts approximately 88.46% of the samples.
* **Precision: 0.8951612903225806**
    * Precision measures the proportion of correctly predicted positive instances (Parkinson's in this case) out of all instances predicted as positive. It answers the question: "Of all the instances predicted as Parkinson's, how many were actually Parkinson's?" A precision of 89.52% indicates that when the model predicts Parkinson's, it's correct about 89.52% of the time.
* **Recall: 0.9568965517241379**
    * Recall (also known as sensitivity or true positive rate) measures the proportion of correctly predicted positive instances out of all actual positive instances. It answers: "Of all the actual Parkinson's cases, how many did the model correctly identify?" A recall of 95.69% means the model captures 95.69% of all actual Parkinson's cases.
* **Specificity: 0.675**
    * Specificity (also known as the true negative rate) measures the proportion of correctly predicted negative instances (healthy cases) out of all actual negative instances. It answers: "Of all the actual healthy cases, how many did the model correctly identify?" A specificity of 67.5% means the model correctly identifies 67.5% of the healthy cases.
* **F1-Score: 0.925**
    * The F1-score is the harmonic mean of precision and recall. It provides a balanced measure of a model's performance, especially when dealing with imbalanced datasets. An F1-score of 0.925 indicates a good balance between precision and recall.
* **Balanced Accuracy: 0.815948275862069**
    * Balanced accuracy is the average of recall and specificity. It's particularly useful when dealing with imbalanced datasets, as it provides a more accurate representation of the model's overall performance. A value of 0.8159 suggests a reasonable performance, considering both positive and negative classes.
* **ROC AUC: 0.9418103448275863**
    * ROC AUC (Receiver Operating Characteristic Area Under the Curve) measures the model's ability to distinguish between positive and negative classes. A value of 0.9418 indicates excellent discrimination ability. A higher ROC AUC indicates that the model is better at distinguishing between the two classes.
* **Average Precision: 0.9779927524166523**
    * Average Precision (AP) summarizes the precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight. A high AP score of 0.978 suggests that the model has very good precision and recall across different thresholds.

## Future Work

Future work on this project could include exploring other machine learning models, such as deep learning models, to further improve the accuracy of the prediction. In addition, the project could be extended to incorporate other types of data, such as genetic data or imaging data, to provide a more comprehensive prediction of Parkinson's disease.

## How to Run

1. Clone this repository to your Google Colab environment.
2. Upload the dataset to your Colab environment.
3. Run the notebook cells to execute the code.

## Requirements

* Python 3.x
* Google Colab
* Required libraries (listed in the notebook)

## Disclaimer

This project is intended for educational and research purposes only. It is not intended to provide medical advice or diagnosis. Please consult with a healthcare professional for any health concerns.
