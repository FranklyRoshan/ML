# Thompson Sampling (Posterior Sampling)

## What It Is
**Thompson Sampling** (also known as the Bayesian Control Rule or Posterior Sampling) is a powerful, stochastic algorithm used to solve the **Exploration vs. Exploitation dilemma** in Multi-Armed Bandit (MAB) problems. 

Unlike the deterministic Upper Confidence Bound (UCB) algorithm—which calculates an artificial mathematical ceiling for each action—Thompson Sampling takes a **Bayesian approach**. It maintains a probability distribution for the expected reward of every action. Each time the algorithm needs to make a choice, it pulls a random sample from each distribution and selects the action that returns the highest sample. 

This elegant framework naturally balances exploration and exploitation: actions with high uncertainty have wide distributions (allowing for wild, exploratory samples), while actions proven to be successful have tight distributions focused around high values (leading to consistent exploitation).

---

# The Core Principle: Bayesian Updating

The beauty of Thompson Sampling lies in how it models its beliefs using a prior distribution, which dynamically morphs into a posterior distribution as real-world data arrives. 

For binary rewards (Success/Failure, Click/No-Click), the algorithm uses the **Beta-Binomial conjugate pair**. The Beta distribution, parameterized by $\alpha$ (Alpha, successes) and $\beta$ (Beta, failures), perfectly maps probabilities between 0 and 1.
```
      Belief Distributions for Two Different Actions (Arms)
Probability
Density
▲
│            Arm A (Highly Exploited / Narrow Variance)
│                 █
│                ███          Arm B (Uncertain / Wide Variance)
│               █████                 ▄█▄
│              ███████             ▄███████▄
└─────────────█████████──────────▄███████████▄──────────► Reward Rate
0.0           └───┬───┘          └─────┬─────┘          1.0
Known to be            Could be poor,
around ~0.30           but might be ~0.80!
```

* **Arm A** has been pulled many times; we are highly confident its true reward rate is around 30%.
* **Arm B** has rarely been pulled. Even though its current average is around 60%, its wide distribution means a random sample could easily land at 80%, triggering an exploratory pull.

---

# The Mathematical Framework

For a Multi-Armed Bandit with Bernoulli rewards, the math follows a clean sequence:

### 1. The Prior
For each action $a$, initialize the tracking parameters to reflect total uncertainty:
$$\alpha_a = 1, \quad \beta_a = 1$$
This creates a uniform distribution where every reward probability from 0 to 1 is equally likely.

### 2. Sampling Step
At each time step $t$, draw a random sample $\theta_a$ from the Beta distribution of each action:
$$\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$$

### 3. Action Selection
Pick the action $A_t$ that generated the highest sampled value:
$$A_t = \arg\max_{a} \theta_a$$

### 4. Posterior Update
Observe the scalar reward $R_t \in \{0, 1\}$ from the environment and update only the chosen action's distribution:
$$\alpha_{A_t} \leftarrow \alpha_{A_t} + R_t$$
$$\beta_{A_t} \leftarrow \beta_{A_t} + (1 - R_t)$$

---

# Algorithmic Mechanics

┌────────────────────────────────────────────────────────┐
│  For each arm, sample θ_a ~ Beta(α_a, β_a)              │
└───────────────────────────┬────────────────────────────┘
▼
┌────────────────────────────────────────────────────────┐
│  Select arm with max θ_a ➔ Pull Arm ➔ Observe Reward   │
└───────────────────────────┬────────────────────────────┘
▼
┌────────────────────────────────────────────────────────┐
│  If Success: α_arm += 1   │   If Failure: β_arm += 1   │
└───────────────────────────┴────────────────────────────┘


Unlike UCB, Thompson Sampling **does not require an initialization pass** where every arm is pulled once. It can start sampling immediately from its uniform priors.

---

# Python Implementation: Bernoulli Multi-Armed Bandit

Below is a complete simulation showing Thompson Sampling adapting to find the optimal slot machine out of three options.

```python
import numpy as np

class BernoulliBanditEnv:
    def __init__(self):
        # Hidden true success probabilities for 3 different actions
        self.true_probabilities = [0.22, 0.80, 0.55]
        
    def pull(self, arm_index):
        """Returns 1 for a success, 0 for a failure."""
        if np.random.rand() < self.true_probabilities[arm_index]:
            return 1
        return 0

# 1. Setup Simulation Parameter Constants
env = BernoulliBanditEnv()
num_arms = len(env.true_probabilities)
total_steps = 1000

# 2. Initialize Bayesian Priors (alpha=1, beta=1 for all arms)
alphas = np.ones(num_arms)
betas = np.ones(num_arms)

# Tracking records for final performance reporting
arm_counts = np.zeros(num_arms)
total_rewards = 0

# 3. Main Thompson Sampling Loop
for t in range(total_steps):
    samples = np.zeros(num_arms)
    
    # Stochastic Sampling Step
    for arm in range(num_arms):
        samples[arm] = np.random.beta(alphas[arm], betas[arm])
        
    # Selection Phase: Choose the arm with the highest sample value
    selected_arm = np.argmax(samples)
    
    # Environment Interaction
    reward = env.pull(selected_arm)
    
    # Posterior Update Phase
    if reward == 1:
        alphas[selected_arm] += 1
    else:
        betas[selected_arm] += 1
        
    # Log keeping
    arm_counts[selected_arm] += 1
    total_rewards += reward

print("--- Thompson Sampling Simulation Results ---")
for i in range(num_arms):
    estimated_mean = alphas[i] / (alphas[i] + betas[i])
    print(f"Arm {i} (True Prob: {env.true_probabilities[i]}): Pulled {int(arm_counts[i])} times. Posterior Mean: {estimated_mean:.4f}")
print(f"\nTotal Cumulative Reward: {total_rewards} out of {total_steps} steps.")
```

# UCB vs. Thompson Sampling: Multi-Armed Bandits & Reinforcement Learning

Both **Upper Confidence Bound (UCB)** and **Thompson Sampling** are elite strategies designed to solve the **Exploration vs. Exploitation dilemma**. While they are technically algorithms for the **Multi-Armed Bandit (MAB)** problem (a simplified, single-state form of Reinforcement Learning), they serve as foundational concepts for exploring unknown environments in full-scale RL.

---

# Key Differences: At a Glance

The core difference lies in their philosophy: **UCB is deterministic and optimistic**, whereas **Thompson Sampling is probabilistic and Bayesian**.

| Feature | Upper Confidence Bound (UCB) | Thompson Sampling |
| :--- | :--- | :--- |
| **Core Philosophy** | Optimism in the face of uncertainty. | Probability matching (Posterior sampling). |
| **Nature** | **Deterministic:** For a given dataset, the chosen action is 100% predictable. | **Stochastic:** Draws random samples from a belief distribution every turn. |
| **Mathematical Basis** | Frequentist (Confidence intervals via Hoeffding's Inequality). | Bayesian (Prior and Posterior updating using probability distributions). |
| **Initialization** | Requires pulling every single arm at least once to initialize variance. | Can start immediately using uniform prior distributions. |
| **Handling Delayed Feedback**| Poor. It relies on immediate sequential updates to adjust its bounds. | Strong. It handles delayed reward updates gracefully due to its probabilistic framework. |

---

# Which Is Best for Reinforcement Learning?

When moving from simple Multi-Armed Bandits to full-scale, multi-state **Reinforcement Learning (RL)**, **Thompson Sampling is generally considered superior and more scalable.**

### Why Thompson Sampling Wins in Deep RL:
1. **The Curse of State Space:** In complex RL environments (like robotics or video games), there are infinite state-action pairs. Building and maintaining rigid frequentist upper bounds (UCB) for billions of states becomes computationally impossible. 
2. **Deep Learning Integration:** Thompson Sampling maps beautifully into deep neural networks through a concept known as **Bootstrapped DQN** or **Deep Exploration**. Instead of keeping exact counts, a neural network can output a distribution over actions, allowing the agent to sample its way through complex environments.
3. **Exploration in Complex Actions:** UCB's exploration bonus can cause highly erratic, unsafe actions in continuous control environments because it aggressively forces the agent to try anything it hasn't seen. Thompson Sampling explores smoothly based on its current level of overall confidence.

---

# Behind the Scenes: How They Move

To visualize how they behave differently given the exact same scenario, consider two slot machines:
* **Arm A:** 10 pulls, 5 wins (50% success rate). We are *somewhat* confident.
* **Arm B:** 2 pulls, 1 win (50% success rate). We are *highly uncertain*.

```
                          UCB Approach
   Deterministic: Calculates a fixed, inflated ceiling.
   
   Arm A: [──── 50% Average ────|── Bonus ──] ➔ UCB Score = 0.72
   Arm B: [──── 50% Average ────|─────────────── Bonus ───────────────] ➔ UCB Score = 0.95 ★ Winner
───────────────────────────────────────────────────────────────────────────

                    Thompson Sampling Approach
   Probabilistic: Draws a random sample from a curve.
   
   Arm A Distribution:      █ 
                          █████     (Narrower curve)
   
   Arm B Distribution:   ▄▄▄███▄▄▄  (Wider curve; sample could land anywhere)
                         ▲
                         └─ Randomly draws samples. Arm B's sample is 
                            more likely to fluctuate wildly and win the turn.
```
* **UCB** calculates a fixed mathematical ceiling for each arm. Because Arm B has been pulled fewer times, its "uncertainty bonus" is massive, forcing the algorithm to choose Arm B deterministically.
* **Thompson Sampling** looks at the underlying bell curves. It draws a random sample from Arm A's narrow distribution and Arm B's wide distribution. Because Arm B's curve is wide, a random sample has a high probability of landing near the top end, causing it to be naturally selected for exploration.