import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# StandardScaler is used to standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
# SVM stand for support vector machine
# used for classification and regression
from sklearn import svm
from sklearn.metrics import accuracy_score


# Load the dataset
df = pd.read_csv("diabetes.csv")

# Analyze the data
df.dropna(inplace=True)  # Drop rows with null values
print(df.head())  # Display the first 5 rows of the dataset
print(df.shape)  # Display the shape of the dataset
print(df.isnull().sum())  # Check for null values in the dataset
print(df.describe())  # Display statistical data of the dataset
print(df['Outcome'].value_counts())  # Count of unique values in the target column

# Separate features and target
X = df.drop(columns='Outcome', axis=1)  # Features
Y = df['Outcome']  # Target

# Dataset is good to go
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Standardizations 
# Standardization is the process of rescaling the features to have a mean of 0 and a standard deviation of 1
# fit is a method that computes the mean and std to be used for later scaling
# Transform is method that performs scaling on the data
# Scaling is x_scaled = (x - mean) / std

scaler = StandardScaler()  # Create a StandardScaler object
scaler.fit(X_train)  # Fit the scaler to the training data
X_train = scaler.transform(X_train)  # Transform the training data
X_test = scaler.transform(X_test)  # Transform the testing data

# Train the model
# SVC stand  for Support Vector Classification
classifier = svm.SVC(kernel='linear')  # Create a SVM model with a linear kernel
classifier.fit(X_train, Y_train)  # Train the model on the training data
# Make predictions
X_train_pred = classifier.predict(X_train)  # Predict the target for the testing data
# Evaluate the model
accuracy = accuracy_score(Y_train, X_train_pred)  # Calculate the accuracy of the model
print("Accuracy:", accuracy)  # Print the accuracy of the model


# Accuracy on testing data
X_test_pred = classifier.predict(X_test)  # Predict the target for the testing data
# Evaluate the model
accuracy = accuracy_score(Y_test, X_test_pred)  # Calculate the accuracy of the model
print("Accuracy on testing data:", accuracy)  # Print the accuracy of the model

 
