# Classification in Machine Learning Overview

## What It Is
Classification is a supervised learning approach where the machine learning model learns from historical data to assign incoming, unseen observations into discrete, predefined categories or classes. Instead of estimating a continuous numerical quantity, a classification model establishes a decision boundary to partition data space into distinct zones.

---

# Classification Terminologies & Mechanics

### 1. Types of Classification Problems
*   **Binary Classification**: Sorting data into exactly two contrasting classes (e.g., `Spam (1)` vs. `Not Spam (0)`, `Fraudulent` vs. `Legitimate`).
*   **Multi-Class Classification**: Sorting data into three or more distinct classes where a sample can only belong to one category (e.g., Classifying an image as a `Cat`, `Dog`, or `Bird`).
*   **Multi-Label Classification**: Assigning multiple labels to a single observation simultaneously (e.g., Tagging a news article as both `Politics` and `Economics`).

### 2. The Decision Boundary
The geometric threshold line, curve, or hyperplane built by the algorithm to separate data classes. When a new sample's features are plotted, its location relative to this decision boundary determines its predicted class label.

---

# Difference Between Regression and Classification

The primary structural distinction lies entirely in the **nature of the target variable ($y$)** you are trying to predict.



| Structural Dimension | Regression | Classification |
| :--- | :--- | :--- |
| **Output Data Type** | Continuous / Quantitative (Infinite decimals/integers) | Discrete / Qualitative (Fixed categories/labels) |
| **Core Algorithmic Goal** | Fit a trend line, curve, or surface directly *through* the data cloud to minimize residual errors. | Find a decision boundary or hyperplane that *separates* data points into distinct neighborhoods. |
| **Target Variable Example** | Predicting a precise **Salary** value (e.g., `₹75,420.50`). | Predicting a **Job Status** category (e.g., `Employed` vs. `Unemployed`). |
| **Output Range** | Infinite (Can theoretically be any numerical value from $-\infty$ to $+\infty$). | Finite (Strictly limited to a specific number of pre-defined buckets). |
| **Core Core Metrics** | $R^2$ Score, Adjusted $R^2$, RMSE, MAE. | Accuracy, Precision, Recall, F1-Score, ROC-AUC. |
| **Key Algorithms** | Multiple Linear Regression, Polynomial Regression, SVR, Decision Tree Regressor. | Logistic Regression, Support Vector Classifiers (SVC), K-Nearest Neighbors (KNN), Naive Bayes. |

---

# The Logistical Bridge: Logistic Regression

A common point of confusion is **Logistic Regression**. Despite having "Regression" in its name, it is fundamentally a **Classification algorithm**. 

*   **The Intuition:** It uses a linear regression engine to calculate a raw continuous score, but then instantly flattens that score using the **Sigmoid Activation Function**:
    $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
*   **The Classification Output:** The Sigmoid function compresses the infinite continuous range down into a strict scale between `0` and `1`, representing the *probability* of a sample belonging to a certain class. By applying a threshold cut-off (e.g., if probability $\ge 0.5$, classify as `Class 1`, else `Class 0`), it converts a continuous math engine into a discrete classification model.
