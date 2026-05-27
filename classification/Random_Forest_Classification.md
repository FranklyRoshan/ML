# Random Forest Classification

## What It Is
A **Random Forest Classifier** is an ensemble supervised machine learning algorithm that constructs a collection of independent decision trees to solve classification problems. It operates on the principle of **Bootstrap Aggregating (Bagging)** and feature randomness. Instead of relying on a single complex decision tree—which is prone to high variance and overfitting—the classifier trains hundreds of diverse trees. Each tree casts a vote for its predicted class, and the forest selects the class with the **majority vote** as the final output.

---

# The Architectural Mechanics

The success of a Random Forest Classifier depends entirely on creating **uncorrelated trees**. If individual trees make independent mistakes, those mistakes cancel out when their votes are aggregated.

### 1. Bootstrapping (Row-Level Sampling)
Given a training dataset with \(N\) rows, the algorithm creates a unique subset for each individual tree by sampling \(N\) rows randomly **with replacement**. 
* This bootstrap process means some training samples appear multiple times in a single tree's subset, while others are omitted.
* On average, **\(36.8\%\) of the data points are excluded** from any single tree's training process. These are termed **Out-Of-Bag (OOB) samples** and are utilized for internal model validation.

### 2. Random Feature Subspaces (Column-Level Sampling)
In a standard decision tree, the algorithm evaluates every available feature to find the best split point. Random Forest alters this mechanic: at every individual split within every tree, the algorithm is restricted to evaluating a **randomly selected subset of features**.
* For a classification task with \(M\) total features, the subset size is typically set to \(\lfloor\sqrt{M}\rfloor\).
* By forcing trees to split on secondary or tertiary features rather than always choosing the dominant predictor, the forest uncovers varied patterns and structures in the data.

### 3. Aggregating via Majority Vote
Once the forest of \(T\) trees is fully trained, predictions for new samples are handled through an aggregation pipeline:
1. The unlabeled data point is passed down every individual tree in the forest.
2. Each tree outputs a class prediction based on its terminal leaf node.
3. The model Tallies the votes. If 85 out of 100 trees vote for Class 1, the model returns **Class 1**.

---

# Mathematics of Forest Classification

While the underlying tree building follows standard split criteria (like Gini Impurity or Entropy), Random Forest introduces mathematical frameworks for evaluation and feature assessment.

### 1. Out-of-Bag (OOB) Error Estimation
Because OOB samples are never seen by a tree during training, they act as a built-in test set. The OOB error is calculated by passing each training row \(x_i\) only through the subset of trees that did *not* use \(x_i\) in their bootstrap sample.

\[\text{OOB Error} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\hat{y}_{\text{OOB}}(x_i) \neq y_i)\]

Where \(\mathbb{I}\) is an indicator function that equals 1 if the OOB majority prediction \(\hat{y}_{\text{OOB}}(x_i)\) does not match the actual label \(y_i\), and 0 otherwise.

### 2. Gini Importance (Mean Decrease in Impurity)
Random Forest calculates how important a feature is by measuring how much the Gini Impurity drops when a split is made using that feature. For a single feature \(X_j\), its importance is summed across all nodes \(C\) where \(X_j\) was selected, and then averaged across all \(T\) trees in the forest:

\[\text{Importance}(X_j) = \frac{1}{T} \sum_{t=1}^{T} \sum_{c \in \text{Nodes}(X_j)} \Delta \text{Gini}(c)\]

Features that frequently sit near the top of trees and create highly pure splits receive a higher importance score.

---

# Python Implementation using Scikit-Learn

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_classes=2, random_state=42)

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and fit the Random Forest Classifier
# n_estimators sets the number of trees; max_features='sqrt' enforces feature randomness
rf_clf = RandomForestClassifier(n_estimators=100, 
                                criterion='gini', 
                                max_features='sqrt', 
                                oob_score=True, 
                                random_state=42)
rf_clf.fit(X_train, y_train)

# 4. Predict and evaluate
y_pred = rf_clf.predict(X_test)

print(f"OOB Validation Accuracy: {rf_clf.oob_score_ * 100:.2f}%")
print(f"Test Set Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))
```

---

# Hyperparameter Tuning

Optimizing a Random Forest Classifier requires tuning parameters that manage both forest diversity and individual tree depth:

### 1. `n_estimators`
The total number of decision trees to build in the forest.
*   **Low values:** Higher risk of underperforming as the voting sample size is too small to cancel out random noise.
*   **High values:** The model becomes structurally stable and protects against overfitting, though it requires more computation time and memory. Performance plateaus after a certain threshold.

### 2. `max_features`
The size of the random subset of features allocated to a node when seeking a split.
*   **Small values (e.g., 1 or 2):** Drastically decreases correlation between trees, but may exclude descriptive features, reducing individual tree accuracy.
*   **Large values (approaching $M$):** Trees look highly similar to one another, causing the ensemble to behave like a standard high-variance decision tree.

### 3. `class_weight`
A tuning parameter used when dealing with imbalanced datasets (e.g., 95% Class 0 and 5% Class 1). Setting this to `"balanced"` automatically adjusts weights inversely proportional to class frequencies, forcing the voting trees to penalize misclassifications of the minority class more severely.
