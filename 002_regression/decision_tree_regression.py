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

# =========================================== DECISION TREE REGRESSION =============================================

# Training the SVR model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

# Visualizing the SVR results ( for higher resolution and smoother curve)
X_grid = np.arange(np.min(X), np.max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# ====================================================== BUG FIX =======================================================

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
