# CART (Classification and Regression Trees) in ML

## What It Is
**CART** (Classification and Regression Trees) is a fundamental, non-parametric machine learning algorithm introduced by Leo Breiman in 1984 that serves as the foundation for modern tree-based methods like Random Forests and Gradient Boosting. Unlike algorithms restricted to a single target type, CART is a unified framework capable of building both **Classification Trees** (to predict discrete categorical classes) and **Regression Trees** (to predict continuous numeric values). It constructs its decision rules by recursively splitting data into exactly two child nodes, producing a strictly **binary tree structure**.

---

# The Architectural Mechanics

While alternative tree algorithms follow multi-way splits, CART adheres strictly to a binary splitting philosophy, isolating data through a sequence of true/false logic gates.

### 1. Binary Splitting Architecture
Every internal decision node in a CART model behaves as a hard binary filter. If a parent node splits on a feature, it creates exactly two child paths. 
*   **Categorical Features:** Splits are structured as subset membership tests (e.g., $\text{Color} \in \{\text{Red}\}$ vs. $\text{Color} \notin \{\text{Red}\}$).
*   **Continuous Features:** Splits are structured as numeric inequality thresholds (e.g., $\text{Income} \leq 50,000$ vs. $\text{Income} > 50,000$).

### 2. The Search for the Optimal Cutpoint
CART uses a greedy, top-down approach called **Recursive Binary Splitting**. The algorithm does not look ahead to see if a split will create a better tree down the line; it focuses purely on immediate optimization. At any given node, it scans every available feature ($X_j$) and every possible threshold value ($s$) for that feature. It evaluates the purity of the data resulting from that specific cutpoint and selects the pair $(X_j, s)$ that maximizes data homogeneity.

### 3. Conceptual Transformation
To visualize this concept clearly: Imagine an automated real estate evaluation system sorting properties into two categories: "Luxury Estates" and "Standard Homes." 

The CART algorithm acts like a series of binary switches. At the root, it evaluates all available data and flips the first switch based on size: *Is the house larger than 4,000 square feet?* Homes that trigger a "No" go down the left track; homes that trigger a "Yes" go down the right track. Down the right track, the algorithm encounters a second binary switch based on location: *Is the house within zip code 90210?* The property continues down these strictly two-way forks until it lands in a terminal leaf box that labels its final market valuation class.

---

# Mathematics of CART Splitting

The mathematical metric CART uses to evaluate the quality of a binary split depends entirely on whether it is constructing a classification tree or a regression tree:

### 1. Classification Metrics (Gini Impurity)
For categorical outcomes, CART defaults to **Gini Impurity** to measure how often a randomly chosen element from the node would be incorrectly labeled. The algorithm seeks the split that causes the largest drop in impurity from the parent node to the weighted sum of the two child nodes.

$$Gini(T) = 1 - \sum_{i=1}^{C} (p_i)^2$$

Where $p_i$ is the probability of a data point belonging to class $i$ within node $T$, and $C$ is the total number of classes.

### 2. Regression Metrics (Squared Residuals / Variance Reduction)
For continuous numeric outcomes, a node cannot be measured by "purity" since the labels are numbers rather than categories. Instead, CART evaluates splits by calculating the **Sum of Squared Errors (SSE)** or **Variance Reduction**. The prediction ($\hat{y}_m$) for a terminal leaf node $m$ is simply the mathematical **mean** of all training outputs falling inside that region.

$$SSE = \sum_{i \in \text{Left}} (y_i - \hat{y}_{\text{Left}})^2 + \sum_{i \in \text{Right}} (y_i - \hat{y}_{\text{Right}})^2$$

The algorithm selects the binary split that results in the lowest possible combined SSE across the two new regions.

---

# Post-Pruning: Cost-Complexity Pruning

Unchecked CART models will split data points down to individual training samples, leading to massive, overfitted trees. While algorithms like ID3 stop early using strict threshold parameters, CART handles overfitting through a robust two-step approach: it allows the tree to grow to its maximum possible size, and then selectively cuts branches back via **Cost-Complexity Pruning** (also known as Weakest Link Pruning).

The algorithm penalizes the complexity of the tree by adding a regularization parameter, $\alpha$ (Alpha), to the cost function:

$$R_\alpha(T) = R(T) + \alpha |T|$$

*   $R(T)$ represents the total misclassification error (or total SSE) of the tree on the training data.
*   $|T|$ represents the total number of terminal leaf nodes (the complexity penalty).
*   **Small $\alpha$:** Keeps the tree large, focusing strictly on minimizing training error.
*   **Large $\alpha$:** Penalizes large trees heavily, forcing branches to collapse and yielding a smaller, highly generalized tree.

---

# Feature Scaling and Missing Data

## 1. Feature Scaling
Just like standard decision trees, CART requires **zero feature scaling** (no Standardization or Normalization). Because every split evaluates a single feature independently using binary logic inequalities, the absolute numeric scale of neighboring variables cannot distort the splitting boundaries.

## 2. Handling Missing Values (Surrogate Splits)
A distinct advantage of the CART algorithm over many machine learning models is its native ability to handle missing data using **Surrogate Splits**. If a data point is missing the value for Feature $A$ at a decision node, CART looks at fallback rules. It calculates which other feature ($Feature\ B$) splits the data most similarly to $Feature\ A$. If $Feature\ B$ is present, the model uses it as a placeholder to safely route the incomplete data point down the correct branch.

---

# Decision Tree Classification

## What It Is
A Decision Tree Classifier is a supervised machine learning algorithm designed to predict categorical outcomes by breaking down a dataset into progressively smaller, more homogeneous subsets. Instead of using complex algebraic equations or geometric hyperplanes, it establishes a hierarchical sequence of simple, nested rules. This structure mimics human decision-making, mapping out a flowchart-like tree where every branch path leads to a specific class prediction.

---

# The Architectural Mechanics

When processing a dataset, a Decision Tree avoids complex global vector calculations. Instead, it systematically splits data space into orthogonal (axis-aligned) rectangular regions.

### 1. The Anatomy of the Tree
The model organizes its logic structure into three distinct structural components:
*   **Root Node:** The peak of the hierarchy. It contains the entire unsplit training dataset and initiates the very first binary rule.
*   **Internal / Decision Nodes:** Intermediate check-points that evaluate a specific feature against a calculated threshold value (e.g., $\text{Age} \leq 30$). Data points meeting the criteria flow down one branch, while the rest flow down the other.
*   **Leaf / Terminal Nodes:** The final endpoints of a branch path. They contain a group of highly similar data points and output the final class prediction based on the majority vote of those points.

### 2. The Mechanics of a Split
A Decision Tree is built top-down using a greedy approach called **Recursive Binary Splitting**. At every single internal node, the algorithm reviews every available feature and evaluates every possible numeric threshold. It calculates the resulting purity of the data if a split were made at that exact point. It selects the single feature and threshold combination that maximizes the purity of the resulting child nodes, splitting the data continuously until a stopping condition is met.

### 3. Conceptual Transformation
To visualize this concept clearly: Imagine a large square field containing a mixed crowd of marathon runners (wearing blue) and casual walkers (wearing red). They are scattered all over the field based on two metrics: Speed ($X$-axis) and Age ($Y$-axis). 

The algorithm acts like a park ranger placing straight fences across the field. First, the ranger drops a vertical fence at the speed mark of 10 km/h. Anyone moving faster than 10 km/h is now trapped on the right side of the fence; this group turns out to be entirely blue marathon runners, creating a perfectly pure zone. On the left side of the fence, the group is still a messy mix of young walkers and elderly runners. The ranger drops a second, horizontal fence across *only* this left section at the Age mark of 60. Now, the field is sliced into neat rectangular corrals. Each corral isolates a distinct group, allowing you to identify anyone inside simply by looking at which fenced box they are standing in.

---

# Mathematics of Tree Splitting

The algorithm determines the optimal placement of its boundaries by minimizing mathematical impurity. The two primary metrics used to calculate this structural purity are:

### 1. Gini Impurity
Gini Impurity measures the probability that a randomly chosen element from a node would be incorrectly classified if it were randomly labeled according to the distribution of classes in that subset. A Gini score of $0$ indicates absolute purity (all records belong to a single class).
$$Gini = 1 - \sum_{i=1}^{C} (p_i)^2$$
Where $p_i$ represents the proportion of data points belonging to class $i$ within that specific node, and $C$ is the total number of distinct classes.

### 2. Entropy and Information Gain
Entropy measures the fundamental level of randomness, disorder, or unpredictability inside a node. Information Gain is the calculated drop in overall entropy achieved by splitting the dataset on a specific feature.
$$Entropy(S) = - \sum_{i=1}^{C} p_i \log_2(p_i)$$
$$Information\ Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)$$
Where $S$ is the current dataset, $A$ is a specific feature, and $S_v$ is the subset of data where feature $A$ takes the value $v$. The algorithm selects the split that yields the highest Information Gain.

---

# Essential Hyperparameter Tuning: Regularization

Unconstrained Decision Trees will grow indefinitely until every single training point is isolated in its own leaf node. This leads to extreme **overfitting**, where the model memorizes training noise. To ensure generalization, you must tune key structural constraints:

### 1. Maximum Depth (`max_depth`)
This parameter controls the maximum vertical distance allowed between the root node and the deepest leaf node.
*   **Small Depth:** Restricts the tree to a few broad splits. The model looks only at macro-trends, creating a simple boundary structure that lowers variance but risks **underfitting** (high bias).
*   **Large Depth:** Allows the tree to grow complex branches. The model creates highly customized paths for tiny clusters of data, capturing micro-anomalies and risking extreme **overfitting** (high variance).

### 2. Minimum Samples Split (`min_samples_split`)
This defines the minimum number of data points that must reside within an internal node before the algorithm is allowed to split it further.
*   **Small Value:** The tree can split nodes containing only two or three points, forcing the model to create highly specific, narrow rules for tiny subsets of data.
*   **Large Value:** Forces the tree to stop splitting early, leaving smaller mixed groups as terminal leaves and keeping the overall tree structure highly generalized.

---

# Feature Scaling in Decision Trees

Feature scaling via **Standardization** or **Normalization** is **completely unnecessary** before training a Decision Tree Classifier.

## The Core Reason: Monotonic Invariance
Decision Tree splits are calculated entirely by evaluating one independent feature at a time using monotonic inequalities (e.g., $X_1 \leq 50$). Because the algorithm looks at the relative distribution ranking of a feature rather than the absolute physical distance between coordinates, the scale of other features does not matter. If Feature 1 is Annual Income (\$30,000 to \$200,000) and Feature 2 is Age (18 to 75), the tree evaluates them completely independently. Multiplying or dividing a feature by a factor of millions shifts the numeric threshold value but leaves the exact position of the split unchanged.
