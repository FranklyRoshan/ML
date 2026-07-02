import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ────────────────────────────────────────────────────────────────
# TRAIN_URL = "https://hrcdn.net/s3_pub/istreet-assets/LgLPfzg0V-7G1vBzJsBxdA/train.csv"
# TEST_URL  = "https://hrcdn.net/s3_pub/istreet-assets/PV13ExA_QndhFEhxoaHG_A/test.csv"
TRAIN_URL = "train.csv"
TEST_URL = "test.csv"

df      = pd.read_csv(TRAIN_URL)
test_df = pd.read_csv(TEST_URL)

# ── Clean: replace non-standard nulls, then drop ─────────────────────────────
df.replace("?", np.nan, inplace=True)
test_df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
test_df.dropna(inplace=True)

# ── Split features / target ──────────────────────────────────────────────────
X = df.drop("income_>50K", axis=1)
y = df["income_>50K"]

# ── Encode categoricals: fit on train, transform both ────────────────────────
cat_cols = X.select_dtypes(include="object").columns

for col in cat_cols:
    le = LabelEncoder()
    X[col]       = le.fit_transform(X[col])           # fit + transform train
    # Handle unseen labels in test gracefully
    test_df[col] = test_df[col].map(
        lambda v, le=le: le.transform([v])[0]
        if v in le.classes_ else -1
    )

# ── Scale ────────────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)                    # fit only on train data

# ── Train / validation split ─────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── Model: class_weight='balanced' handles the 33k vs 10k imbalance ──────────
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ── Retrain on full data, predict test set ───────────────────────────────────
model.fit(X_scaled, y)

test_scaled = scaler.transform(test_df)               # transform, not fit
test_df["income_50K"] = model.predict(test_scaled)

test_df[["income_50K"]].to_csv("submission.csv", index=True)
print("Saved submission.csv")

# --------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ────────────────────────────────────────────────────────────────
TRAIN_URL = "https://hrcdn.net/s3_pub/istreet-assets/LgLPfzg0V-7G1vBzJsBxdA/train.csv"
TEST_URL  = "https://hrcdn.net/s3_pub/istreet-assets/PV13ExA_QndhFEhxoaHG_A/test.csv"

df      = pd.read_csv(TRAIN_URL)
test_df = pd.read_csv(TEST_URL)

# ── Clean: replace non-standard nulls, then impute with mode ─────────────────
df.replace("?", np.nan, inplace=True)
test_df.replace("?", np.nan, inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in test_df.select_dtypes(include="object").columns:
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)

# ── Split features / target ──────────────────────────────────────────────────
X = df.drop("income_>50K", axis=1)
y = df["income_>50K"]

# ── Feature engineering ──────────────────────────────────────────────────────
X["capital_net"] = X["capital-gain"] - X["capital-loss"]
test_df["capital_net"] = test_df["capital-gain"] - test_df["capital-loss"]

X["age_bucket"] = pd.cut(X["age"], bins=[0, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4]).astype(int)
test_df["age_bucket"] = pd.cut(test_df["age"], bins=[0, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4]).astype(int)

# ── Encode categoricals: fit on train, transform both ────────────────────────
cat_cols = X.select_dtypes(include="object").columns

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    # Handle unseen labels in test gracefully
    test_df[col] = test_df[col].map(
        lambda v, le=le: le.transform([v])[0]
        if v in le.classes_ else -1
    )

# ── Align test columns to train (same columns, same order) ───────────────────
test_features = test_df[X.columns]

# ── Train / validation split ─────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model: Random Forest handles non-linearity + imbalance ───────────────────
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1        # use all CPU cores
)
model.fit(X_train, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ── Feature importance plot ──────────────────────────────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind="bar")
plt.title("Top 10 Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# ── Retrain on full data, predict test set ───────────────────────────────────
model.fit(X, y)

test_df["income_50K"] = model.predict(test_features)
test_df[["income_50K"]].to_csv("submission.csv", index=True)
print("Saved submission.csv")

# ------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load data ────────────────────────────────────────────────────────────────
TRAIN_URL = "https://hrcdn.net/s3_pub/istreet-assets/LgLPfzg0V-7G1vBzJsBxdA/train.csv"
TEST_URL  = "https://hrcdn.net/s3_pub/istreet-assets/PV13ExA_QndhFEhxoaHG_A/test.csv"

df      = pd.read_csv(TRAIN_URL)
test_df = pd.read_csv(TEST_URL)

# ── Clean: replace non-standard nulls ────────────────────────────────────────
df.replace("?", np.nan, inplace=True)
test_df.replace("?", np.nan, inplace=True)

# ── Impute: mode for categoricals, median for numerics ───────────────────────
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in test_df.select_dtypes(include="object").columns:
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include="number").columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in test_df.select_dtypes(include="number").columns:
    test_df[col].fillna(test_df[col].median(), inplace=True)

# ── Split features / target ──────────────────────────────────────────────────
X = df.drop("income_>50K", axis=1).copy()   # .copy() prevents SettingWithCopyWarning
y = df["income_>50K"]

# ── Feature engineering ──────────────────────────────────────────────────────
X["capital_net"]       = X["capital-gain"] - X["capital-loss"]
test_df["capital_net"] = test_df["capital-gain"] - test_df["capital-loss"]

X["age_bucket"]       = pd.cut(X["age"], bins=[0, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4]).astype(int)
test_df["age_bucket"] = pd.cut(test_df["age"], bins=[0, 25, 35, 50, 65, 100], labels=[0, 1, 2, 3, 4]).astype(int)

# ── Encode categoricals: fit on train, transform both ────────────────────────
cat_cols = X.select_dtypes(include="object").columns

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    # Handle unseen labels in test gracefully
    test_df[col] = test_df[col].map(
        lambda v, le=le: le.transform([v])[0]
        if v in le.classes_ else -1
    )

# ── Align test columns to train (same columns, same order) ───────────────────
test_features = test_df[X.columns]

# ── Train / validation split ─────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,          # prevents overfitting
    min_samples_leaf=5,    # smooths the tree
    class_weight="balanced",
    random_state=42,
    n_jobs=-1              # use all CPU cores
)
model.fit(X_train, y_train)

# ── Evaluate: check for overfitting ──────────────────────────────────────────
train_acc = accuracy_score(y_train, model.predict(X_train))
val_acc   = accuracy_score(y_val,   model.predict(X_val))

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Val Accuracy:   {val_acc:.4f}")
if train_acc - val_acc > 0.05:
    print("⚠️  Possible overfitting — consider lowering max_depth")
else:
    print("✅  Train/val gap looks healthy")

print("\n", classification_report(y_val, model.predict(X_val)))

# ── Confusion matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_val, model.predict(X_val))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

# ── Feature importance plot ───────────────────────────────────────────────────
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind="bar")
plt.title("Top 10 Feature Importances")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

# ── Retrain on full data, predict test set ───────────────────────────────────
model.fit(X, y)

test_df["income_50K"] = model.predict(test_features)
test_df[["income_50K"]].to_csv("submission.csv", index=True)
print("Saved submission.csv")