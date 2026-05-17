# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('pima-indians-diabetes.csv')

# Identify missing data (assumes that missing data is represented as NaN)
X = df.iloc[:, :-1].values
y = df.iloc[:,  -1].values

# Print the number of missing entries in each column
print(X)
print(y)

# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame
imputer.fit(X[:, 1:3])

# Apply the transform to the DataFrame
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Print your updated matrix of features
print(X)