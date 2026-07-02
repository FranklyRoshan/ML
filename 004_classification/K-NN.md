# K-Nearest Neighbors (KNN) Overview

## What It Is
K-Nearest Neighbors (KNN) is a simple, non-parametric, and supervised machine learning algorithm used for both **Classification** and **Regression** tasks. Primarily utilized for classification, it operates on a fundamentally intuitive premise: similar data points exist in close proximity to one another within the multi-dimensional feature space. 

---

# The Architectural Mechanics

Unlike algorithms that learn explicit mathematical parameters (like weights or coefficients) from training data, KNN is an **Instance-Based** and **Lazy Learner**. It performs zero training computations, instead memorizing the training dataset intact and executing all logic during inference.

### 1. The Proximity Computation (The Distance Metric)
When a new unlabeled query point ($x$) is introduced, the algorithm computes the geometric distance between $x$ and every single saved sample ($y$) in the training library across $n$ features. The standard structural metrics include:

*   **Euclidean Distance (L2 Norm):** Measures straight-line distance.
    $$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

*   **Manhattan Distance (L1 Norm):** Measures absolute grid-like distance.
    $$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

### 2. Neighbor Extraction
Once all distances are calculated, the model sorts the distances in ascending order. It then isolates the absolute top **$K$** training observations that sit closest to the query point.

### 3. The Visual Spatial Boundary
The variable $K$ functions as a structural radius boundary that encloses the nearest data points around the test sample. 

In a visual two-dimensional feature space where Feature 1 forms the horizontal axis and Feature 2 forms the vertical axis, data points belonging to Class A (marked as O) cluster in the upper-left area, while data points belonging to Class B (marked as X) cluster in the lower-right area. Right in the middle sits a new, unlabeled Query Point (marked as ?). The algorithm draws a circular neighborhood boundary around this target query point to evaluate which group of neighbors falls closest to it.

### 4. The Majority Voting Consensus
To output a definitive classification prediction, the algorithm tabulates the class labels of the extracted $K$ neighbors. The label with the highest tally wins:

$$\hat{y} = \operatorname{mode}(y_1, y_2, \dots, y_K)$$

*   If $K=3$ and the neighbors are `[Class A, Class B, Class A]`, the prediction is **Class A**.
*   **Tie-Breaking:** If an even $K$ value results in a split vote, ties are broken arbitrarily, by prioritizing the closest overall neighbor, or by applying distance-weighting.

---

# Core Assumptions
1.  **Distance Equals Similarity:** The fundamental assumption is that features clustered closely in space share identical semantic target classes.
2.  **Clean Feature Space:** The dataset is assumed to have minimal noise, as arbitrary outliers drastically distort local neighborhood boundaries.
3.  **Uniform Feature Relevance:** It assumes all dimensions contribute equally to the distance calculation.

---

# Feature Scaling in KNN

Feature scaling (Standardization or MinMax Scaling) is **strictly mandatory** for K-Nearest Neighbors.

## The Core Reason: Distance Dominance
Because KNN evaluates similarity purely through geometric distance metrics, the absolute scale of the input features directly controls the calculations. 
*   **The Scale Skew:** If one feature spans a massive range (e.g., Annual Income: \$20,000 to \$500,000) and another spans a narrow range (e.g., Age: 18 to 65), the distance formula will completely ignore the Age feature. 
*   **The Fix:** Scaling squashes all input features into identical numerical bounds (like 0 to 1 or a mean of 0 and variance of 1), ensuring every dimension exerts equal leverage over the final classification vote.
