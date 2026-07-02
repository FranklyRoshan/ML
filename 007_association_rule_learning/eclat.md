# The ECLAT Algorithm (Equivalence Class Transformation)

## What It Is
The **ECLAT Algorithm** is a popular, unsupervised data mining algorithm used to discover frequent itemsets in transactional databases. While its counterpart, the Apriori algorithm, scans the dataset horizontally, ECLAT completely flips the paradigm by adopting a **vertical data format**. 

Instead of reading a long list of transactions to see what items are inside them, ECLAT maps each unique product to a list of every single transaction ID (TID) where it appears. By converting the database into this vertical alignment, finding frequent itemsets becomes a blazing-fast operation of basic set mathematics—completely eliminating the need to repeatedly scan a massive hard drive.

---

# The Core Mechanism: Horizontal vs. Vertical Layouts

The foundational difference between Apriori and ECLAT lies in how the data is structured and processed in memory:

### Horizontal Data Format (Apriori)
Data is stored as a sequence of transaction records.

* TID 1: {Milk, Bread, Diapers} 
* TID 2: {Milk, Diapers, Beer} 
* TID 3: {Bread, Beef}


### Vertical Data Format (ECLAT)
Data is inverted into an item-centric layout called a **TID-set** (Transaction ID set).

* Milk:    {1, 2} 
* Bread:   {1, 3} 
* Diapers: {1, 2} 
* Beer:    {2} 
* Beef:    {3}


---

# How ECLAT Works: The Power of Set Intersection

ECLAT utilizes a **depth-first search (DFS)** strategy to explore the itemset lattice. To find out how frequently a combination of items occurs, it does not scan rows; it simply intersects the corresponding TID-sets.

$$\text{TID-set}(X \cup Y) = \text{TID-set}(X) \cap \text{TID-set}(Y)$$

The **Support** of the combined itemset is nothing more than the count (cardinality) of the resulting intersected set.

### Step-by-Step Execution Example:
Imagine you want to calculate the support for the itemset **{Milk, Diapers}**:

1. Retrieve the TID-set for **Milk**: `{1, 2}`
2. Retrieve the TID-set for **Diapers**: `{1, 2}`
3. Perform a mathematical intersection ($\cap$):
   $$\{1, 2\} \cap \{1, 2\} = \{1, 2\}$$
4. Count the elements: The size is 2. Therefore, the absolute support for `{Milk, Diapers}` is **2**.



Now, if you want to expand that search to a 3-itemset like **{Milk, Diapers, Bread}**, you take your newly created `{Milk, Diapers}` TID-set and intersect it with **Bread** (`{1, 3}`):
$$\{1, 2\} \cap \{1, 3\} = \{1\}$$
The count is 1. The algorithm continues this depth-first recursive intersection pattern until it runs out of combinations that satisfy the minimum support threshold.

---

# Python Implementation using PyEclat

Because standard library packages like `mlxtend` natively focus on horizontal algorithms like Apriori, specialized libraries or custom matrices are used to run ECLAT effectively in Python.

```python
import pandas as pd
from pyEclat import Example_dataset, ECLAT

# 1. Load an example transactional dataset
# Creating a dummy dataframe layout matching transaction history
data = {
    0: ['Milk', 'Bread', 'Diapers', None],
    1: ['Milk', 'Diapers', 'Beer', None],
    2: ['Bread', 'Beef', None, None],
    3: ['Milk', 'Bread', 'Diapers', 'Beer']
}
df = pd.DataFrame.from_dict(data, orient='index')

print("--- Raw Transaction DataFrame ---")
print(df, "\n")

# 2. Initialize the ECLAT Instance
eclat_instance = ECLAT(data=df, verbose=True)

# 3. Look at the Vertical Binary Matrix generated under the hood
print("--- Internal Vertical Binary Representation ---")
print(eclat_instance.df_bin.astype(int), "\n")

# 4. Fit the algorithm 
# min_support = 50% (itemset must appear in at least 2 out of 4 transactions)
# max_combination = 3 (search up to 3-item combinations)
get_ECLAT_indexes, get_ECLAT_supports = eclat_instance.fit(
    min_support=0.5,
    min_combination=1,
    max_combination=3,
    separator=' & '
)

print("\n--- Identified Frequent Itemsets & Support ---")
for itemset, support in sorted(get_ECLAT_supports.items(), key=lambda x: x[1], reverse=True):
    print(f"Itemset: {itemset:<25} | Support: {support * 100:.1f}%")
```