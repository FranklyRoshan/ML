# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ================================================ DATA PREPROCESSING ==================================================

# Importing the dataset (df - DataFrame)
df = pd.read_csv('Data.csv')
#  slice operator [start:stop:step]
X = df.iloc[:, :-1].values # matrix of features (Independent variable vector)
y = df.iloc[:,-1].values # Dependent variable vector

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
