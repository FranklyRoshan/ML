# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading the Iris dataset
df = pd.read_csv('iris.csv')

# Creating the matrix of features (X) and the dependent variable vector (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Printing the matrix of features and the dependent variable vector
print(X)
print(y)