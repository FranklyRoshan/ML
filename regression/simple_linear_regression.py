# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================== DATA PREPROCESSING ===============================================
# Importing the dataset (df - DataFrame)
df = pd.read_csv('Salary-Data.csv')
#  slice operator [start:stop:step]
X = df.iloc[:, :-1].values # matrix of features (Independent variable vector)
y = df.iloc[:,-1].values # Dependent variable vector

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# =============================================== REGRESSION ===============================================
# Training the Simple Learn Regression model on the  Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualizing the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs. Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Salary vs. Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# =============================================== FAQs ===============================================

# Question 1: How do I use my simple linear regression model to make a single prediction, for example, to predict the
# salary of an employee with 12 years of experience?

# Solution: Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))

# Therefore, our model predicts that the salary of an employee with 12 years of experience is $ 138967,5.
# Important note: Notice that the value of the feature (12 years) was input in a double pair of square brackets.
# That('s because the "predict" method always expects a 2D array as the format of its inputs. And putting 12 into a
# double pair of square brackets makes the input exactly a 2D array. Simply put:)

# 12     -> Scalar
# [12]   -> 1D Array
# [[12]] -> 2D Array

# Question 2: How do I get the final regression equation y = b0 + b1 x with the final values of the
# coefficients b0 and b1?

# Solution: Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)

# Therefore, the equation of our simple linear regression model is:

# Salary = 9345.94×YearsExperience + 26816.19

# Important Note: To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object.
# Attributes in Python are different than methods and usually return a simple value or an array of values.