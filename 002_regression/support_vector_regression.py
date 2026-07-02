# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ================================================ DATA PREPROCESSING ==================================================

# Importing the dataset (df - DataFrame)
df = pd.read_csv('Position_Salaries.csv')
#  slice operator [start:stop:step]
X = df.iloc[:, 1:-1].values # matrix of features (Independent variable vector)
y = df.iloc[:,-1].values # Dependent variable vector
print(X)
print(y)

# list to 2D Array
y = y.reshape(len(y), 1)
print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# ============================================ SUPPORT VECTOR REGRESSION ===============================================

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # Radial Basis Function (RBF) / Gaussian Kernel
regressor.fit(X, y.ravel())

# Predicting a new result
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

# Visualizing the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1,1)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the SVR results ( for higher resolution and smoother curve)
X_grid = np.arange(np.min(sc_X.inverse_transform(X)), np.max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# ====================================================== BUG FIX =======================================================

"""
## Fixing DataConversionWarning in Scikit-Learn
## The Problem
When fitting a Scikit-Learn model (such as SVR or Linear Regression), passing a 2D column vector (shape (n_samples, 1)) 
instead of a flat 1D array (shape (n_samples,)) for your target variable y triggers the following warning:
DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to 
(n_samples, ), for example using ravel(). While this is a warning and your code may still run, it can slow down 
computation and lead to unpredictable matrix behaviors in advanced models.

## The Corrected Code
To fix this, apply the .ravel() method or .flatten() to your target variable y right when you fit your model:

import numpy as npfrom sklearn.svm import SVR
# Assuming X is your 2D feature matrix and y is your 2D column vector
regressor = SVR(kernel='rbf')
# FIX: Use y.ravel() to flatten the target array into 1D
regressor.fit(X, y.ravel())

## Why This Fix Works

* Shape Differences: If you read your data using Pandas (df.iloc[:, -1].values) or applied StandardScaler to your target 
variable y earlier in an SVR workflow, the data structure transforms into a 2D column matrix ([[y1], [y2], [y3]]).
* Scikit-Learn Expectation: The optimization algorithms inside Scikit-Learn expect the target array y to be a simple, 
flat 1D sequence of numbers ([y1, y2, y3]).
* The .ravel() Function: This NumPy function flattens the multi-dimensional array into a continuous 1D array without 
making an unnecessary copy of the data in memory, resolving the warning instantly.

"""
# ----------------------------------------------------------------------------------------------------------------------
"""
## Fixing TypeError in Polynomial Regression Plotting

## The Problem
When plotting a smooth curve for Polynomial Regression, using Python's built-in min(X) and max(X) on a 2D feature matrix 
X throws the following error:
TypeError: only 0-dimensional arrays can be converted to Python scalars

## The Corrected Code

import numpy as np
import matplotlib.pyplot as plt

# FIX: Use np.min(sc_X.inverse_transform(X)) and np.max(sc_X.inverse_transform(X)) instead of 
# min(sc_X.inverse_transform(X)) and max(sc_X.inverse_transform(X)) 
np.arange(np.min(sc_X.inverse_transform(X)), np.max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
# Plotting the results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Why This Fix Works

* Shape Mismatch: Your feature matrix X was likely transformed into a 2D array via .reshape(-1, 1) earlier in your code 
so it could be fed into Scikit-Learn.
* Built-in Functions: Python's native min() expects a 1D iterable sequence. When it encounters a 2D array, it fails to 
compress the extra dimension into a single scalar.
* NumPy Functions: np.min() and np.max() are designed to scan across multi-dimensional arrays, automatically flattening 
them to find the absolute minimum and maximum boundary values safely.

"""
# ----------------------------------------------------------------------------------------------------------------------
