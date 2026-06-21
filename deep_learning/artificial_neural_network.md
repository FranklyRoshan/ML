# Artificial Neural Networks: A High-Dimensional Topography

An **Artificial Neural Network (ANN)** can be conceptualized as a highly parallel, continuous mathematical mapping system rather than an organic brain. It is an optimization engine designed to approximate an unknown function $f^*$ by mapping an input vector $\mathbf{x}$ to a predicted target vector $\mathbf{\hat{y}}$ through a series of geometric coordinate transformations.

---

# 1. Topological Decomposition

An ANN operates as an assembled pipeline of distinct structural phases. Each phase acts as a tensor processor, shifting data dimensions to isolate latent patterns.


```
[X: Input Tensor]         [H: Latent Feature Space]         [Y: Output Target]
Coordinate Features         Non-Linear Projections           Final Estimations

 ┌───────────┐                ┌───────────┐                ┌───────────┐
 │    X₁     ├───────────────►│    H₁     ├───────────────►│    Y₁     │
 ├───────────┤ \            / ├───────────┤ \            / ├───────────┤
 │    X₂     │  ▀►─────────▀  │    H₂     │  ▀►─────────▀  │    Y₂     │
 ├───────────┤ /            \ ├───────────┤ /            \ └───────────┘
 │    X₃     ├───────────────►│    H₃     ├─┘
 └───────────┘                └───────────┘
```

* **The Input Space ($\mathbf{X}$):** Represents the raw feature matrix. It defines the initial vector space dimensionality ($d_{\text{in}}$).
* **The Hidden Space ($\mathbf{H}$):** A manifold where the network warps, stretches, and rotates the input coordinates. This processing isolates linearly inseparable distributions into clean, distinct clusters.
* **The Output Space ($\mathbf{\hat{y}}$):** The final structural contraction. It maps the learned features down to target configurations, generating continuous values for regression or logit boundaries for classification.

---

# 2. Mathematical Vector Synthesis

The progression of data through a single node within the network relies on affine transformations coupled with non-linear mapping filters:

$$\mathbf{h} = \sigma \left( \mathbf{X}\mathbf{W} + \mathbf{b} \right)$$

### Matrix Component Breakdown:
* $\mathbf{X} \in \mathbb{R}^{M \times d_{\text{in}}}$: The incoming feature array for a batch of $M$ samples.
* $\mathbf{W} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$: The **Weight Matrix**. This acts as a transformation operator, governing the rotation and scaling of the data vector into the next space.
* $\mathbf{b} \in \mathbb{R}^{1 \times d_{\text{out}}}$: The **Bias Vector**. This introduces a spatial translation, allowing the transformation boundary to shift away from the origin.
* $\sigma$: The **Activation Function** (e.g., $\text{ReLU}(z) = \max(0, z)$). This acts as a non-linear gate. Without this step, consecutive matrix multiplications would mathematically collapse into a single linear equation, making the network incapable of learning complex, non-linear structures.

---

# 3. Dimensional Tensor Tracking

When engineering network architectures, monitoring tensor shapes across layers prevents runtime compilation errors. The table below outlines how a batch of data changes shape as it moves through a single hidden-layer system:

| Phase | Mathematical Operation | Tensor Dimensions |
| :--- | :--- | :--- |
| **Input Feature Feed** | $\mathbf{X}$ | $\text{Batch Size } (M) \times \text{Input Features } (3)$ |
| **First Space Transformation** | $\mathbf{Z}^{(1)} = \mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}$ | $(M \times 3) \cdot (3 \times 3) + (1 \times 3) \rightarrow \mathbf{M \times 3}$ |
| **Non-Linear Filtering** | $\mathbf{H} = \max(0, \mathbf{Z}^{(1)})$ | Dimension conserved: $\mathbf{M \times 3}$ |
| **Output Space Projection** | $\mathbf{Z}^{(2)} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$ | $(M \times 3) \cdot (3 \times 2) + (1 \times 2) \rightarrow \mathbf{M \times 2}$ |

---

# 4. Pure Functional Implementation (Vectorized NumPy Engine)

Below is an objective object-oriented blueprint of a two-layer ANN built from scratch using matrix calculus. It demonstrates the structural forward pass and explicit backpropagation routines.

```python
import numpy as np

class TensorNeuralNetwork:
    def __init__(self, d_in, d_hidden, d_out, seed=42):
        np.random.seed(seed)
        # Initialize weight matrices using standard normal scaling
        self.W1 = np.random.randn(d_in, d_hidden) * 0.1
        self.b1 = np.zeros((1, d_hidden))
        self.W2 = np.random.randn(d_hidden, d_out) * 0.1
        self.b2 = np.zeros((1, d_out))
        
    def _sigmoid(self, Z):
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
        
    def _sigmoid_derivative(self, A):
        # Local gradient calculation given active state A
        return A * (1.0 - A)

    def forward(self, X):
        """Executes spatial transformations across the network layers."""
        self.X = X
        # Project input matrix to hidden space
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.H = self._sigmoid(self.Z1)
        
        # Project hidden space to output layer
        self.Z2 = np.dot(self.H, self.W2) + self.b2
        self.Y_hat = self._sigmoid(self.Z2)
        return self.Y_hat

    def backward(self, Y, learning_rate):
        """Applies the calculus Chain Rule to calculate error gradients and update weights."""
        m = self.X.shape[0]
        
        # 1. Output error mapping
        dZ2 = (self.Y_hat - Y) * self._sigmoid_derivative(self.Y_hat)
        
        # 2. Backpropagate error to the hidden space
        dH = np.dot(dZ2, self.W2.T)
        dZ1 = dH * self._sigmoid_derivative(self.H)
        
        # 3. Calculate exact partial derivatives (Gradients)
        dW2 = np.dot(self.H.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dW1 = np.dot(self.X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # 4. Update the parameter space using Gradient Descent
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# --- Execution Verification ---
if __name__ == "__main__":
    # Generate an analytical non-linear dataset (XOR Logic Gate Topology)
    X_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([[0],   [1],   [1],   [0]])

    # Initialize Engine: 2 Inputs -> 3 Hidden Dimensions -> 1 Output Target
    ann = TensorNeuralNetwork(d_in=2, d_hidden=3, d_out=1)
    
    # Run optimization loop across parameter space
    for epoch in range(10000):
        predictions = ann.forward(X_train)
        ann.backward(y_train, learning_rate=0.3)
        
    print("--- Final Model Approximations ---")
    final_inference = ann.forward(X_train)
    for i in range(len(X_train)):
        print(f"Features: {X_train[i]} | Target: {y_train[i][0]} | Inference: {final_inference[i][0]:.4f}")
```