# ============================================================
# ML LAB - MOBILE PRICE CLASSIFICATION
# Cross Validation with Visualization
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    ShuffleSplit,
    cross_val_score
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

# ============================================================
# 2. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")
print(data.isnull().sum())

print("\nColumns:")
print(data.columns.tolist())

# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

# price_range is the target column
X = data.drop(columns=["price_range"])
y = data["price_range"]

print("\nNumber of features:", X.shape[1])
print("Number of samples:", X.shape[0])

print("\nPrice Range Distribution:")
print(y.value_counts().sort_index())

# ============================================================
# 4. CREATE MACHINE LEARNING MODEL
# ============================================================

model = DecisionTreeClassifier(
    random_state=42
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

model.fit(X, y)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(10, 7))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Decision Tree Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# ============================================================
# VISUALIZATION 1
# PRICE RANGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    x="price_range",
    data=data
)

plt.title("Mobile Price Range Distribution")
plt.xlabel("Price Range")
plt.ylabel("Number of Mobile Phones")

plt.tight_layout()
plt.show()

# ============================================================
# VISUALIZATION 2
# RAM VS PRICE RANGE
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="price_range",
    y="ram",
    data=data
)

plt.title("RAM vs Mobile Price Range")
plt.xlabel("Price Range")
plt.ylabel("RAM")

plt.tight_layout()
plt.show()

# ============================================================
# VISUALIZATION 3
# BATTERY POWER VS PRICE RANGE
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="price_range",
    y="battery_power",
    data=data
)

plt.title("Battery Power vs Mobile Price Range")
plt.xlabel("Price Range")
plt.ylabel("Battery Power")

plt.tight_layout()
plt.show()

# ============================================================
# 5. K-FOLD CROSS VALIDATION
# ============================================================

print("\n======================================")
print("1. K-FOLD CROSS VALIDATION")
print("======================================")

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

kfold_scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="accuracy"
)

for i, score in enumerate(kfold_scores, start=1):
    print(f"Fold {i} Accuracy: {score:.4f}")

kfold_mean = kfold_scores.mean()

print(f"\nAverage K-Fold Accuracy: {kfold_mean:.4f}")
print(f"Average K-Fold Accuracy: {kfold_mean * 100:.2f}%")

# ============================================================
# 6. STRATIFIED K-FOLD CROSS VALIDATION
# ============================================================

print("\n======================================")
print("2. STRATIFIED K-FOLD CROSS VALIDATION")
print("======================================")

stratified = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

stratified_scores = cross_val_score(
    model,
    X,
    y,
    cv=stratified,
    scoring="accuracy"
)

for i, score in enumerate(stratified_scores, start=1):
    print(f"Fold {i} Accuracy: {score:.4f}")

stratified_mean = stratified_scores.mean()

print(
    f"\nAverage Stratified K-Fold Accuracy: "
    f"{stratified_mean:.4f}"
)

print(
    f"Average Stratified K-Fold Accuracy: "
    f"{stratified_mean * 100:.2f}%"
)

# ============================================================
# 7. LEAVE-ONE-OUT CROSS VALIDATION
# ============================================================

print("\n======================================")
print("3. LEAVE-ONE-OUT CROSS VALIDATION")
print("======================================")

loo = LeaveOneOut()

loo_scores = cross_val_score(
    model,
    X,
    y,
    cv=loo,
    scoring="accuracy"
)

loo_mean = loo_scores.mean()

print("Total LOOCV Tests:", len(loo_scores))

print(f"LOOCV Accuracy: {loo_mean:.4f}")
print(f"LOOCV Accuracy: {loo_mean * 100:.2f}%")

# ============================================================
# 8. SHUFFLE SPLIT CROSS VALIDATION
# ============================================================

print("\n======================================")
print("4. SHUFFLE SPLIT CROSS VALIDATION")
print("======================================")

shuffle_split = ShuffleSplit(
    n_splits=5,
    test_size=0.2,
    random_state=42
)

shuffle_scores = cross_val_score(
    model,
    X,
    y,
    cv=shuffle_split,
    scoring="accuracy"
)

for i, score in enumerate(shuffle_scores, start=1):
    print(f"Split {i} Accuracy: {score:.4f}")

shuffle_mean = shuffle_scores.mean()

print(
    f"\nAverage Shuffle Split Accuracy: "
    f"{shuffle_mean:.4f}"
)

print(
    f"Average Shuffle Split Accuracy: "
    f"{shuffle_mean * 100:.2f}%"
)

# ============================================================
# 9. COMPARE ALL CROSS-VALIDATION METHODS
# ============================================================

results = pd.DataFrame({
    "Method": [
        "K-Fold",
        "Stratified K-Fold",
        "Leave-One-Out",
        "Shuffle Split"
    ],
    "Accuracy": [
        kfold_mean,
        stratified_mean,
        loo_mean,
        shuffle_mean
    ]
})

print("\n======================================")
print("CROSS VALIDATION RESULTS")
print("======================================")

print(results.to_string(index=False))

# ============================================================
# 10. ACCURACY COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(
    results["Method"],
    results["Accuracy"]
)

plt.title("Cross Validation Accuracy Comparison")
plt.xlabel("Cross Validation Method")
plt.ylabel("Accuracy")

plt.ylim(
    0,
    1
)

plt.xticks(rotation=15)

# Display accuracy on top of each bar
for bar, value in zip(bars, results["Accuracy"]):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value * 100:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()

# ============================================================
# 11. FINAL RESULT
# ============================================================

print("\n======================================")
print("FINAL RESULT")
print("======================================")

for _, row in results.iterrows():

    print(
        f"{row['Method']}: "
        f"{row['Accuracy'] * 100:.2f}%"
    )

print("\nExperiment completed successfully!")