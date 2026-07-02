# Logistic Regression Overview

## What It Is
Logistic Regression is a supervised learning algorithm used to predict the probability of a discrete target variable ($y$) belonging to a specific class. Despite having "Regression" in its name, it is a foundational **Classification algorithm** used primarily for binary outcomes (e.g., Yes/No, Spam/Not Spam, Default/No Default).

---

# The Architectural Mechanics

Instead of fitting a straight line through data points like Linear Regression, Logistic Regression maps an S-shaped curve (the **Sigmoid Function**) to express the relationship between input features and class probabilities.

### 1. The Linear Core (The Log-Odds)
The algorithm starts by calculating a standard linear combination of your input features, identical to Multiple Linear Regression:
$$z = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n$$

### 2. The Activation Bridge (The Sigmoid Function)
To turn that infinite linear score ($z$) into a stable probability value between `0` and `1`, the algorithm passes $z$ through the mathematical **Sigmoid (Logistic) Function**:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

*   **$p$**: The output probability that the given observation belongs to the positive class (Class 1).
*   **$e$**: Euler's constant (~2.718).
*   **$z$**: The raw input value from your linear combinations.

### 3. The Visual S-Curve Behavior
*   If $z$ becomes a **large positive number**, $e^{-z}$ approaches $0$, forcing the probability $p$ to climb toward **$1$**.
*   If $z$ becomes a **large negative number**, $e^{-z}$ becomes massive, forcing the probability $p$ to drop toward **$0$**.
*   If $z = 0$, $e^{0} = 1$, which makes the probability $p$ exactly **$0.5$** (the dead center baseline).

```
 Probability (p)
    1.0 |               .---''
        |             .'
    0.5 |-----------.'----------- (Decision Threshold Boundary)
        |         .'
    0.0 |___..--''
        +-----------------------
                 z (Log-Odds Score)
```

### 4. The Decision Threshold Boundary
To make a final discrete classification choice, the model checks the calculated probability against a designated threshold value (the default default is **0.5**):
*   If $p \ge 0.5 \rightarrow$ Predict **Class 1** (e.g., Transaction is Fraudulent).
*   If $p < 0.5 \rightarrow$ Predict **Class 0** (e.g., Transaction is Normal).

---

# Core Assumptions
1.  **Binary Outcome:** The target output vector ($y$) must be discrete and dichotomous (two classes).
2.  **Linearity in Log-Odds:** The relationship between the independent predictor features and the *log-odds* of the target variable must be strictly linear.
3.  **Independence of Observations:** Data rows must be completely independent of one another.
4.  **No Multicollinearity:** Independent input features ($X$) must not be highly correlated with each other.

---

# Feature Scaling in Logistic Regression

Feature scaling (like Standardization) is **highly recommended**, especially if you utilize regularization loops.

## The Core Reason: Optimization Speed and Regularization
*   **Optimization (Gradient Descent):** If your model uses solvers like `gradient descent` or `sag/saga` to minimize the cost function, widely different feature scales (e.g., Age 0-100 vs. Annual Income 0-1,000,000) create elongated cost valleys, slowing down convergence.
*   **Regularization Penalties:** By default, Scikit-Learn's `LogisticRegression` applies **L2 (Ridge) regularization** penalty boundaries. If features sit on different scales, the model will unfairly penalize the weights of small-scale variables.
