# Activation Functions in Deep Learning

## What They Are
In an Artificial Neural Network, an **Activation Function** is a mathematical equation attached to each neuron. Its primary responsibility is to introduce **non-linearity** into the network's mathematical transformations. 

Without non-linearity, no matter how many hundreds of hidden layers you stack, successive matrix multiplications will always collapse into a simple, single-layer linear regression ($y = wx + b$). Non-linear activation functions allow neural networks to learn highly complex, curved decision boundaries and approximate any arbitrary, continuous function.

---

# The Core Mathematical Transformation

When data flows into a neuron, it first undergoes a linear affine transformation (weighted sum + bias), and the result $Z$ is passed directly into the activation function $\sigma$:

$$A = \sigma(Z) = \sigma\left( \sum_{i=1}^{n} w_i x_i + b \right)$$



---

# Primary Activation Functions

Modern architectures utilize different activation functions depending on whether they are placed within the **Hidden Layers** (extracting features) or the **Output Layer** (formatting final predictions).

### 1. ReLU (Rectified Linear Unit)
The industry standard for hidden layers in deep networks. It is computationally incredibly cheap and mitigates the vanishing gradient problem.
* **Formula:** 
$$f(x) = \max(0, x)$$
* **Range:** $[0, \infty)$
* **Pros/Cons:** Eliminates vanishing gradients for positive inputs. However, it can suffer from the **"Dying ReLU"** problem, where neurons getting negative inputs output exactly 0, completely stopping backpropagation gradient flow through them forever.

### 2. Leaky ReLU
An evolution of ReLU designed specifically to fix the Dying ReLU issue.
* **Formula:** 
$$f(x) = \max(\alpha x, x) \quad \text{where } \alpha \approx 0.01$$
* **Range:** $(-\infty, \infty)$
* **Pros/Cons:** Ensures that even if a neuron outputs a negative value, it still passes a small local gradient ($\alpha$) backward during optimization.

### 3. Sigmoid
Historically popular, now primarily restricted to the final output layer for binary classification tasks.
* **Formula:** 
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
* **Range:** $(0, 1)$
* **Pros/Cons:** Maps values cleanly to probabilities. However, for highly positive or negative inputs, the curve becomes completely flat, driving the derivative to near zero. This causes **Vanishing Gradients**, stalling model learning.

### 4. Softmax
The mathematical standard for the output layer of multi-class classification problems.
* **Formula:** 
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$
* **Range:** $(0, 1)$, where the sum of all outputs equals exactly 1.0.
* **Pros/Cons:** Turns an unnormalized vector of raw model scores (logits) into a true probability distribution over multiple distinct classes.

---

# Architectural Comparison Matrix

| Function | Plot Topology | Best Used In | Primary Threat |
| :--- | :--- | :--- | :--- |
| **ReLU** | Flat for $x<0$, linear for $x>0$ | Hidden Layers (CNNs, MLPs) | Dead Neurons |
| **Leaky ReLU** | Slight slope for $x<0$, linear for $x>0$ | Hidden Layers (GANs, Deep Networks) | Hyperparameter tuning ($\alpha$) |
| **Sigmoid** | S-shaped curve | Output Layer (Binary Classification) | Vanishing Gradient |
| **Softmax** | Exponential normalizer | Output Layer (Multi-Class Classification) | Exploding logits |

---

# Python Implementation: Tensors & Gradients in PyTorch

Below is a verification script mapping how different activation functions alter tensor elements and calculate local derivatives.

```python
import torch
import torch.nn as nn

# 1. Create a mock tensor simulating raw neuron outputs (logits)
# Contains negative, zero, and positive values
raw_logits = torch.tensor([-2.0, 0.0, 4.0], requires_grad=True)

# 2. Instantiate Activations
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.1)
sigmoid = nn.Sigmoid()
softmax = nn.Softmax(dim=0)

print("--- Forward Pass Structural Modifications ---")
print(f"Raw Input Logits : {raw_logits.detach().numpy()}")
print(f"ReLU Output      : {relu(raw_logits).detach().numpy()}")
print(f"Leaky ReLU (0.1) : {leaky_relu(raw_logits).detach().numpy()}")
print(f"Sigmoid Output   : {sigmoid(raw_logits).detach().numpy()}")
print(f"Softmax Outputs  : {softmax(raw_logits).detach().numpy()} -> Sum: {softmax(raw_logits).sum().item():.1f}\n")

print("--- Backpropagation Gradient Verification (ReLU vs Leaky ReLU) ---")
# Reset and evaluate ReLU gradient flow on negative values
output_relu = relu(raw_logits)
output_relu.backward(torch.ones_like(raw_logits))
print(f"ReLU Local Gradients       : {raw_logits.grad.numpy()}  <- (0.0 means dead node)")

# Reset gradients
raw_logits.grad.zero_attr = None 
raw_logits.grad = None

# Evaluate Leaky ReLU gradient flow on negative values
output_lrelu = leaky_relu(raw_logits)
output_lrelu.backward(torch.ones_like(raw_logits))
print(f"Leaky ReLU Local Gradients : {raw_logits.grad.numpy()}  <- (Preserved active path)")