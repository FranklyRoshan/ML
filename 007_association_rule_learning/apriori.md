# The Apriori Algorithm

## What It Is
The **Apriori Algorithm** is a foundational, unsupervised data mining algorithm designed to identify frequent itemsets within a transactional database to establish association rules. It solves a classic combinatorial optimization problem: if a store sells 10,000 unique products, evaluating every possible combination of items to see what sells together would require checking billions of variations—a process that would quickly crash a standard server.

Apriori systematically prunes this massive search space by leveraging a core mathematical property: **all subsets of a frequent itemset must also be frequent.** If a subset is rare, any larger combination containing it is instantly discarded without being checked, allowing companies to discover valuable customer behavior patterns efficiently.

---

# The Core Principle: Apriori Downward Closure

The entire efficiency of the algorithm rests on the **Apriori Property** (also known as the downward-closure property of support):

$$\text{If } I \in A \text{ and } \text{Support}(A) < \text{min\_support}, \text{ then } \text{Support}(I) \le \text{Support}(A) < \text{min\_support}$$



### The Monotonicity Rule:
* **Down Pass (Pruning):** If an itemset (e.g., `{Beer}`) is found to be infrequent (below your defined minimum support threshold), then any larger itemset containing it (e.g., `{Beer, Diapers}`) is **guaranteed** to be infrequent. There is absolutely no reason to waste computational resources calculating its frequency.
* **Up Pass (Joining):** To find frequent itemsets of size $k+1$, you only combine itemsets of size $k$ that have already passed the minimum support filter.

---

# Step-by-Step Algorithmic Mechanics

The algorithm functions as an iterative, level-wise search. It alternates between a **Join step** (generating candidate larger sets) and a **Prune step** (removing non-viable candidates based on data scans).

---
```
[Database Scan 1] ➔ Count Candidate 1-itemsets (C1) ➔ Prune ➔ Frequent 1-itemsets (L1)
                                                                 │
┌────────────────────────────────────────────────────────────────┘
▼
[Join L1 with itself] ➔ Candidate 2-itemsets (C2) ➔ [Database Scan 2] ➔ Prune ➔ Frequent 2-itemsets (L2)
                                                                                 │
┌────────────────────────────────────────────────────────────────────────────────┘
▼
[Join L2 with itself] ➔ Candidate 3-itemsets (C3) ➔ ... (Repeat until no new frequent itemsets are found)
```

### 1. Initialization ($k=1$)
* Scan the database once to calculate the support of every single individual item.
* Filter out all items that fall below the `min_support` threshold. The remaining items form the set of Frequent 1-itemsets ($L_1$).

### 2. Candidate Generation & Joining ($k \rightarrow k+1$)
* Take the frequent itemsets of size $k$ ($L_k$) and join them with each other to create candidate itemsets of size $k+1$ ($C_{k+1}$).
* **The Join Condition:** Two itemsets are joined only if their first $k-1$ items are identical.

### 3. Pruning
* For every candidate in $C_{k+1}$, check if any of its internal sub-itemsets of size $k$ are missing from your previous frequent list ($L_k$).
* If a sub-component is missing, instantly delete that entire candidate from $C_{k+1}$.

### 4. Database Verification
* Scan the actual transaction database to count the true occurrences of the remaining candidates in $C_{k+1}$.
* Keep only the ones that meet the `min_support` threshold to create the new Frequent ($L_{k+1}$) list.
* Repeat Steps 2–4 until no new frequent itemsets can be generated.

---

# Python Implementation using Scikit-Learn Ecosystem

```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Raw Transaction History
transactions = [
    ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
    ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice Cream', 'Eggs']
]

# 2. Transform raw text lists into a structural boolean matrix
te = TransactionEncoder()
te_matrix = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_matrix, columns=te.columns_)

# 3. Apply the Apriori Algorithm (Set minimum support to 60%)
# This means an itemset must appear in at least 3 out of 5 transactions
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

print("--- Identified Frequent Itemsets ---")
print(frequent_itemsets.sort_values(by='support', ascending=False).to_string(index=False))
print("\n")

# 4. Generate Association Rules from the frequent itemsets (Set minimum confidence to 80%)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

# Clean up output display columns
cleaned_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print("--- Derived Rules Passing Confidence Threshold ---")
print(cleaned_rules.sort_values(by='lift', ascending=False).to_string(index=False))