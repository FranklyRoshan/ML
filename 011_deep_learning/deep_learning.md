# Artificial Neural Networks (ANN)

## What It Is
An **Artificial Neural Network (ANN)** is a computational model inspired by the biological structure and functioning of the human brain. It forms the foundational backbone of deep learning. An ANN consists of interconnected processing units called **neurons** (or nodes) that work in parallel to learn patterns, pass signals, and map complex relationships between raw inputs and target outputs.

Unlike traditional software that relies on explicit, hardcoded logic blocks, an ANN learns how to perform tasks by analyzing training examples—essentially adjusting its internal connections dynamically based on its errors.

---

# The Structural Architecture

A standard Artificial Neural Network arranges its neurons into a distinct sequence of layers. Signals flow forward through these layers in a process known as data propagation.

        Anatomy of an Artificial Neural Network
```
Input Layer        Hidden Layer         Output Layer
(Data Features)   (Pattern Extraction)    (Predictions)

[INPUT SPACE]               [HIDDEN SPACE]               [OUTPUT SPACE]
3-Dimensional               3-Dimensional                2-Dimensional

┌─────────┐   Matrix W¹     ┌─────────┐   Matrix W²     ┌─────────┐
│   X₁    │  ───────────►   │   H₁    │  ───────────►   │   Y₁    │
│   X₂    │   (3x3 Grid     │   H₂    │   (3x2 Grid     │   Y₂    │
│   X₃    │   Rotation)     │   H₃    │  Projection)    └─────────┘
└─────────┘                 └─────────┘
```

### The 3 Structural Layers:
1. **Input Layer:** Receives raw features from the dataset (e.g., age, income, and credit score for a financial model). It performs no mathematical operations; it simply passes the numbers along.
2. **Hidden Layer(s):** The network's processing engine. Neurons in this layer combine inputs from the previous layer to extract hidden features, abstract representations, and non-linear patterns. A network can have one hidden layer (shallow) or hundreds (deep).
3. **Output Layer:** Translates the hidden representations into final usable predictions, such as a continuous scalar value (Regression) or a set of class probabilities (Classification).

---

# Mathematical Mechanics of a Single Neuron

To understand how an entire network operates, consider the mathematical transformation happening inside every individual neuron. A neuron accepts input values, computes an aggregated score, and maps it to an output.

$$y = \sigma \left( \sum_{i=1}^{n} w_i x_i + b \right)$$

### Breaking Down the Components:
* $x_i$: The input features or signals arriving from the preceding layer.
* $w_i$ (**Weights**): The relative strength or importance assigned to each input signal. If a specific feature is highly predictive, its weight grows larger.
* $b$ (**Bias**): An adjustable offset parameter that shifts the activation function along its horizontal axis. It allows the neuron to output values even when all inputs are zero.
* $\sigma$ (**Activation Function**): A non-linear mathematical function (e.g., **ReLU**, **Sigmoid**, or **Tanh**) applied to the weighted sum. Without this non-linearity, multiple layers would mathematically collapse into a single linear equation, rendering the network unable to learn complex patterns like curves or intersecting bounds.

---

# The Learning Loop: Training the Network

An ANN learns iteratively through a continuous feedback loop divided into three main operational phases:

### 1. The Forward Pass
Inputs are passed into the network, travel through the hidden layers via the weight matrices, and emerge from the output layer as a prediction ($\hat{y}$).

### 2. Loss Computation
The network evaluates how accurate its guess was by comparing $\hat{y}$ to the actual ground-truth target ($y$) using a designated **Loss Function** (e.g., Mean Squared Error for regression). The goal of training is to drive this loss score as close to zero as possible.

### 3. Backpropagation & Optimization
This is where learning actually occurs:
* **The Gradient:** The network uses the calculus **Chain Rule** to compute the gradient of the loss function. It tracks backward through the network, determining exactly how much credit or blame each weight and bias deserves for the final error.
* **The Update:** An optimization algorithm (like Gradient Descent or Adam) updates every weight in the network by stepping in the opposite direction of the calculated gradient, scaled by a hyperparameter called the **Learning Rate** ($\alpha$).

$$w \leftarrow w - \alpha \frac{\partial \text{Loss}}{\partial w}$$

---

# Python Implementation: Building a Raw ANN from Scratch

Below is a complete implementation of a 2-layer Artificial Neural Network constructed purely using NumPy to showcase the explicit mathematics behind the forward pass and backpropagation.

```python
import numpy as np

class ArtificialNeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        # Seed random number generator for reproducibility
        np.random.seed(42)
        
        # Initialize Weights and Biases with small random values
        self.W1 = np.random.randn(input_nodes, hidden_nodes) * 0.1
        self.b1 = np.zeros((1, hidden_nodes))
        
        self.W2 = np.random.randn(hidden_nodes, output_nodes) * 0.1
        self.b2 = np.zeros((1, output_nodes))
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def _sigmoid_derivative(self, x):
        # x is already the sigmoid output
        return x * (1 - x)

    def forward(self, X):
        """Propagates inputs forward through the 2-layer network."""
        # Layer 1: Hidden Layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._sigmoid(self.Z1)
        
        # Layer 2: Output Layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self._sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y, output, learning_rate):
        """Calculates gradients and updates internal weights and biases."""
        m = X.shape[0] # Number of training samples
        
        # 1. Calculate Error at the Output Layer
        error_output = output - y
        delta_output = error_output * self._sigmoid_derivative(output)
        
        # 2. Backpropagate error to the Hidden Layer
        error_hidden = np.dot(delta_output, self.W2.T)
        delta_hidden = error_hidden * self._sigmoid_derivative(self.A1)
        
        # 3. Compute partial derivatives (Gradients)
        dW2 = np.dot(self.A1.T, delta_output) / m
        db2 = np.sum(delta_output, axis=0, keepdims=True) / m
        
        dW1 = np.dot(X.T, delta_hidden) / m
        db1 = np.sum(delta_hidden, axis=0, keepdims=True) / m
        
        # 4. Update Weights and Biases (Gradient Descent Step)
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

# --- Test the Implementation ---
if __name__ == "__main__":
    # Create an XOR logic gate dataset (Classical non-linear problem)
    X_train = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_train = np.array([[0],   [[1]],  [[1]],  [[0]]]) # Fixed dimensionality for calculation
    y_train = y_train.reshape(4, 1)

    # Initialize network: 2 Inputs -> 3 Hidden Neurons -> 1 Output Prediction
    ann = ArtificialNeuralNetwork(input_nodes=2, hidden_nodes=3, output_nodes=1)
    
    print("Training the custom ANN over 5000 epochs...")
    for epoch in range(5000):
        predictions = ann.forward(X_train)
        ann.backward(X_train, y_train, predictions, learning_rate=0.5)
        
    print("\n--- Final Predictions after Training ---")
    final_outputs = ann.forward(X_train)
    for i in range(len(X_train)):
        print(f"Input: {X_train[i]} | Targeted Target: {y_train[i][0]} | Model Prediction: {final_outputs[i][0]:.4f}")
```

