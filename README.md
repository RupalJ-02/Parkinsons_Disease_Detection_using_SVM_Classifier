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

The trained SVM model achieved an accuracy of [insert accuracy here]% on the test set. This result demonstrates the potential of using voice measurements for the early detection and prediction of Parkinson's disease.

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
