# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset (df - DataFrame)
df = pd.read_csv('Data.csv')
#  slice operator [start:stop:step]
X = df.iloc[:, :-1].values # matrix of features (Independent variable vector)
y = df.iloc[:,-1].values # Dependent variable vector
# print(X)
# print(y)

# taking care of missing data
"""
Replacing missing values with the column mean is a standard technique in data preprocessing.
"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encode categorial data
# Encoding the independent variable vector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X = np.array(ct.fit_transform(X))
# print(X)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)

# Splitting the dataset into the Training set and Test set
"""
To prevent Overfitting a situation where a Machine Learning model learns 
only on one dataset and cannot adapt to any other.
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# Feature Scaling
"""
Scaling column values so that they are comparable
"""
# Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass