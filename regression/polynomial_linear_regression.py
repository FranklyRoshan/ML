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

# =========================================== POLYNOMIAL LINEAR REGRESSION =============================================

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)

X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing the Linear Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(np.min(X), np.max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_pred = lin_reg.predict([[6.5]])
print(lin_pred)

# Predicting a new result with Polynomial Regression
poly_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_pred)

# ====================================================== BUG FIX =======================================================

"""
## Fixing TypeError in Polynomial Regression Plotting
## The Problem
When plotting a smooth curve for Polynomial Regression, using Python's built-in min(X) and max(X) on a 2D feature 
matrix X throws the following error:
TypeError: only 0-dimensional arrays can be converted to Python scalars

## The Corrected Code

import numpy as np
import matplotlib.pyplot as plt

# FIX: Use np.min(X) and np.max(X) instead of min(X) and max(X)X_grid 
np.arange(np.min(X), np.max(X), 0.1)X_grid 
X_grid.reshape((len(X_grid), 1))
# Plotting the results
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Why This Fix Works

* Shape Mismatch: Your feature matrix X was likely transformed into a 2D array via .reshape(-1, 1) earlier in your 
code so it could be fed into Scikit-Learn.
* Built-in Functions: Python's native min() expects a 1D iterable sequence. When it encounters a 2D array, it fails to 
compress the extra dimension into a single scalar.
* NumPy Functions: np.min() and np.max() are designed to scan across multi-dimensional arrays, automatically flattening 
them to find the absolute minimum and maximum boundary values safely.

"""
# ----------------------------------------------------------------------------------------------------------------------
