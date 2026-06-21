# The Upper Confidence Bound (UCB) Algorithm

## What It Is
The **Upper Confidence Bound (UCB)** algorithm is a deterministic, mathematically grounded approach to solving the **Exploration vs. Exploitation dilemma** in Multi-Armed Bandit (MAB) problems. 

Instead of randomly guessing to explore (like the $\epsilon$-greedy approach), UCB operates on a core philosophy: **Optimism in the Face of Uncertainty**. It assumes that an arm with highly uncertain payouts *could* be spectacularly good. Therefore, it systematically favors arms that either have a high known success rate (exploitation) or haven't been tested enough yet (exploration).

---

# The Core Principle: Quantifying Uncertainty

Standard algorithms look only at the average historical reward of an action. UCB builds a **confidence interval** around that average. 

As an action is chosen more often, our uncertainty about its true reward decreases, and the confidence interval shrinks. If an action is ignored, the confidence interval remains wide, meaning its "upper bound" stays high, making it attractive to the algorithm again.

True Reward Distribution & Confidence Intervals
```
Reward
▲
│          [─── Upper Confidence Bound (UCB) ───]
│                         ▲
│                         │ Uncertainty / Exploration Bonus
│                         ▼
│                   ┌───────────┐
│                   │  μ̂ (s,a)  │ ◄── Historical Average (Exploitation)
│                   └───────────┘
│                         ▲
│                         │
│                         ▼
│          [────────────────────────────────────]
└────────────────────────────────────────────────────────► Actions
```
---

# The Mathematical Framework: UCB1

The most widely used variant, **UCB1**, selects the action $A_t$ at time step $t$ that maximizes the following formula:

$$A_t = \arg\max_{a} \left( Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right)$$

### Breaking Down the Components:
1. **The Exploitation Term ($Q_t(a)$):** The estimated (average) reward of action $a$ up to time step $t$. It encourages choosing actions that have performed well in the past.
2. **The Exploration Term ($\sqrt{\frac{\ln t}{N_t(a)}}$):** This measures the variance or uncertainty of action $a$. 
   * $t$ is the total number of rounds played across all actions. The natural log ($\ln t$) means uncertainty grows slowly over time for skipped actions.
   * $N_t(a)$ is the number of times action $a$ has been selected. The more you pick it, the larger the denominator gets, and the exploration term shrinks toward zero.
3. **The Exploration Constant ($c$):** A user-defined hyperparameter (theoretically $c=\sqrt{2}$ in original literature) that controls how heavily the algorithm weighs uncertainty against historical success.

---

# Algorithmic Mechanics

UCB1 requires a clean, structured lifecycle loop to dynamically adjust its bounds.

### 1. Initialization
* Play every available arm/action exactly **once** ($N(a) = 1$ for all $a$).
* Record the initial rewards and set $t$ equal to the total number of actions. This guarantees that $N_t(a)$ is never zero, preventing a division-by-zero error.

### 2. The Selection Phase
* Calculate the metric $Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}}$ for every available arm.
* Pull the arm with the highest total value.

### 3. The Update Phase
* Observe the actual reward $R_t$ from the environment.
* Update the historical count for that specific arm: $N(a) \leftarrow N(a) + 1$.
* Recalculate the running mean reward $Q(a)$ for that arm.
* Advance the global timer: $t \leftarrow t + 1$.
* Loop back to Step 2.

---

# Python Implementation: Multi-Armed Bandit Simulation

Below is a clean simulation comparing UCB against a multi-armed bandit environment with 3 distinct reward rates.

```python
import numpy as np
import math

class BanditEnvironment:
    def __init__(self):
        # True hidden win probabilities for 3 different slot machines
        self.true_probabilities = [0.30, 0.75, 0.45]
        
    def pull(self, arm_index):
        """Returns a binary reward (1 for win, 0 for loss) based on true prob."""
        if np.random.rand() < self.true_probabilities[arm_index]:
            return 1.0
        return 0.0

# 1. Setup Simulation Constants
env = BanditEnvironment()
num_arms = len(env.true_probabilities)
total_steps = 1000
c = 1.5  # Exploration parameter

# 2. Tracking Variables
arm_counts = np.zeros(num_arms)    # N_t(a)
arm_rewards = np.zeros(num_arms)   # Total scalar reward per arm
q_values = np.zeros(num_arms)      # Q_t(a) - running average reward

# 3. Initialization Step: Pull every arm exactly once
for arm in range(num_arms):
    reward = env.pull(arm)
    arm_counts[arm] += 1
    arm_rewards[arm] += reward
    q_values[arm] = arm_rewards[arm] / arm_counts[arm]

# 4. Main UCB Production Loop
for t in range(num_arms, total_steps):
    ucb_values = np.zeros(num_arms)
    
    for arm in range(num_arms):
        # Standard UCB1 Formula
        exploration_bonus = c * math.sqrt(math.log(t) / arm_counts[arm])
        ucb_values[arm] = q_values[arm] + exploration_bonus
        
    # Select the arm with the highest Upper Confidence Bound
    selected_arm = np.argmax(ucb_values)
    
    # Interact with environment
    reward = env.pull(selected_arm)
    
    # Update tracking records
    arm_counts[selected_arm] += 1
    arm_rewards[selected_arm] += reward
    q_values[selected_arm] = arm_rewards[selected_arm] / arm_counts[selected_arm]

print("--- Simulation Metrics ---")
for i in range(num_arms):
    print(f"Arm {i} (True Prob: {env.true_probabilities[i]}): Pulled {int(arm_counts[i])} times. Estimated Q-Value: {q_values[i]:.4f}")