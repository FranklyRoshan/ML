# Reinforcement Learning (RL)

## What It Is
**Reinforcement Learning (RL)** is a branch of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize the notion of cumulative reward. Unlike supervised learning (which learns from a labeled training set provided by a supervisor) or unsupervised learning (which finds hidden structure in unlabeled data), RL relies entirely on a **trial-and-error** feedback loop.

Imagine training a dog: you don't give it a manual on how to sit (supervised data). Instead, it tries different behaviors, and when it sits on command, you give it a treat (positive reinforcement). Over time, the dog associates the command and the action with a reward. 

---

# The Core Framework: Agent-Environment Loop

The fundamental framework of reinforcement learning is modeled as a **Markov Decision Process (MDP)**. The process is a continuous loop of interactions between two main components:

*   **The Agent:** The AI system, decision-maker, or learner.
*   **The Environment:** The world or system the agent interacts with (e.g., a chess board, a video game screen, or a stock market).

```
              ┌───────────────┐
              │  Environment  │
              └───────┬───────┘
                      │
        Reward ($R_t$) │ State ($S_t$)
                      ▼
              ┌───────────────┐
              │     Agent     │
              └───────┬───────┘
                      │
                      │ Action ($A_t$)
                      ▼
```

### The 4 Pillars of the Loop
1.  **State ($S_t$):** The current snapshot or observation of the environment at time step $t$.
2.  **Action ($A_t$):** The move, choice, or decision the agent makes from the available options.
3.  **Reward ($R_{t+1}$):** The scalar feedback returned by the environment evaluates the action (can be positive, negative, or zero).
4.  **Next State ($S_{t+1}$):** The new situation the agent finds itself in as a direct result of its action.

---

# Key Concepts & Mathematics

To move from conceptual loops to functional algorithms, RL relies on mathematical foundations to calculate long-term value over short-term gratification.

### 1. The Return ($G_t$)
An agent's objective is to maximize the cumulative expected return. Because future rewards are less certain and less valuable than immediate rewards, RL uses a **discount factor** ($\gamma$, gamma, where $0 \le \gamma \le 1$).

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

### 2. Policy ($\pi$)
The policy is the agent's brain—a mapping from perceived states of the environment to actions to be taken. 
*   **Deterministic:** $\pi(s) = a$ (Always choose action $a$ in state $s$)
*   **Stochastic:** $\pi(a|s) = P(A_t = a | S_t = s)$ (The probability of choosing action $a$ given state $s$)

### 3. Value Functions
Value functions estimate how good it is for the agent to be in a given state, or how good it is to perform a specific action in a given state.
*   **State-Value Function ($V^\pi(s)$):** Expected return starting from state $s$ and following policy $\pi$ thereafter.
*   **Action-Value Function ($Q^\pi(s, a)$):** Expected return starting from state $s$, taking action $a$, and then following policy $\pi$ thereafter. 

### 4. The Exploration vs. Exploitation Dilemma
This is a core challenge in RL:
*   **Exploitation:** The agent chooses the best-known action to maximize immediate reward based on its current knowledge.
*   **Exploration:** The agent tries novel, unmapped actions to discover potentially better strategies for the long term.
> **Note:** A common approach to balance this is the **$\epsilon$-greedy (epsilon-greedy)** strategy, where the agent explores with probability $\epsilon$ and exploits with probability $1-\epsilon$.

---

# Major RL Algorithms

RL algorithms are generally split into **Model-Free** (learning purely from experience) and **Model-Based** (building or using a simulation of the environment). Within Model-Free, we have two primary paradigms:

| Category | Description | Examples |
| :--- | :--- | :--- |
| **Value-Based** | Learns to estimate the optimal $Q$-value for every state-action pair; chooses the action with the highest value. | Q-Learning, Deep Q-Networks (DQN) |
| **Policy-Based** | Optimizes the policy ($\pi$) directly without relying heavily on a value function. Excellent for continuous actions. | REINFORCE, Policy Gradient |
| **Actor-Critic** | Hybrid approach. The **Actor** updates the policy based on feedback, while the **Critic** evaluates the action via a value function. | PPO (Proximal Policy Optimization), SAC, A3C |

---

# Python Implementation: Standard Q-Learning

Below is a simple demonstration of tabular **Q-Learning** solving a custom grid-world environment using basic Python constructs.

```python
import numpy as np
import random

# 1. Environment Setup (Simple 4-state line: 0, 1, 2, 3. State 3 is the Goal)
NUM_STATES = 4
NUM_ACTIONS = 2  # 0: Left, 1: Right
GOAL_STATE = 3

# Hyperparameters
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.2  # Exploration rate
episodes = 100

# 2. Initialize Q-Table with zeros
q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

def step(state, action):
    """Simulates environment transition and rewards."""
    if action == 1:  # Move Right
        next_state = min(state + 1, NUM_STATES - 1)
    else:           # Move Left
        next_state = max(state - 1, 0)
        
    reward = 10 if next_state == GOAL_STATE else -1
    return next_state, reward

# 3. Training Loop
for episode in range(episodes):
    state = 0  # Reset to start state
    while state != GOAL_STATE:
        # Action selection (Epsilon-Greedy)
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1])  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
            
        # Environment step
        next_state, reward = step(state, action)
        
        # Temporal Difference (TD) Update Equation
        # Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state, best_next_action]
        q_table[state, action] += alpha * (td_target - q_table[state, action])
        
        state = next_state

print("--- Trained Q-Table ---")
print("States (0 to 3) x Actions (Left, Right)")
print(q_table)
```