# BAN6430_Module_5

Breast Cancer Classification using PCA and Logistic Regression
This project demonstrates how to perform breast cancer classification using Principal Component Analysis (PCA) for dimensionality reduction and Logistic Regression for prediction. The dataset used is the breast cancer dataset available in the scikit-learn library.

Overview
Breast cancer is one of the most common cancers among women worldwide. Early detection and accurate diagnosis are crucial for effective treatment. In this project, we utilize PCA to reduce the dimensionality of the breast cancer dataset and then apply logistic regression to classify tumors as benign or malignant.

Instructions
Requirements
Google colab
**Required Python libraries:** 
numpy
pandas
matplotlib
seaborn
scikit-learn

Setup
Ensure you have a google account which is free but up can pay for an upgraded version
Signup to Google colab by Google which has runtype of Python and R
Ensure all the libraries below are imported to Collab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

Load Dataset: Load the Breast Cancer dataset from sklearn.datasets.
Standardize Features: Standardize the features using StandardScaler.
Perform PCA: Reduce the dimensionality of the dataset using PCA.
Plot PCA Components: Visualize the first two principal components.
Cumulative Explained Variance: Plot the cumulative explained variance to determine the number of components to retain.
Logistic Regression: Train a logistic regression model using the PCA components.
Hyperparameter Tuning: Optimize the logistic regression model using grid search.
Confusion Matrix: Evaluate the model performance using a confusion matrix.


Push the code to Github
