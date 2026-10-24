# ============================================================
# ML LAB - PRINCIPAL COMPONENT ANALYSIS (PCA)
# Mobile Price Classification Dataset
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["price_range"])
y = data["price_range"]

print("\nNumber of original features:", X.shape[1])
print("Number of samples:", X.shape[0])

# ============================================================
# 3. STANDARDIZE THE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData standardized successfully!")

# ============================================================
# 4. APPLY PCA
# ============================================================

# Reduce 20 features to 2 principal components

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA applied successfully!")

print("Original number of features:", X.shape[1])
print("Reduced number of features:", X_pca.shape[1])

# ============================================================
# 5. EXPLAINED VARIANCE
# ============================================================

print("\nExplained Variance Ratio:")

print(
    "PC1:",
    pca.explained_variance_ratio_[0]
)

print(
    "PC2:",
    pca.explained_variance_ratio_[1]
)

total_variance = pca.explained_variance_ratio_.sum()

print(
    "\nTotal Variance Explained by PC1 + PC2:",
    total_variance
)

print(
    "Percentage of Variance Explained:",
    total_variance * 100,
    "%"
)

# ============================================================
# 6. CREATE PCA DATAFRAME
# ============================================================

pca_data = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_data["price_range"] = y.values

print("\nPCA Data:")
print(pca_data.head())

# ============================================================
# GRAPH 1 - EXPLAINED VARIANCE
# ============================================================

plt.figure(figsize=(7, 5))

components = ["PC1", "PC2"]

variance = pca.explained_variance_ratio_

plt.bar(
    components,
    variance
)

plt.title("Explained Variance by Principal Components")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")

for i, value in enumerate(variance):
    plt.text(
        i,
        value,
        f"{value * 100:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()

# ============================================================
# GRAPH 2 - PCA SCATTER PLOT
# ============================================================

plt.figure(figsize=(9, 6))

for price_class in sorted(y.unique()):

    points = pca_data[
        pca_data["price_range"] == price_class
    ]

    plt.scatter(
        points["PC1"],
        points["PC2"],
        label=f"Price Range {price_class}",
        alpha=0.6
    )

plt.title("PCA: Mobile Price Classification")
plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# GRAPH 3 - CUMULATIVE EXPLAINED VARIANCE
# ============================================================

# PCA with all features to calculate cumulative variance

pca_full = PCA()

X_pca_full = pca_full.fit_transform(X_scaled)

cumulative_variance = (
    pca_full.explained_variance_ratio_.cumsum()
)

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)

plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")

plt.axhline(
    y=0.90,
    linestyle="--",
    label="90% Variance"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 7. FINAL RESULT
# ============================================================

print("\n======================================")
print("FINAL PCA RESULT")
print("======================================")

print("Original Features :", X.shape[1])
print("PCA Components    :", X_pca.shape[1])

print(
    f"PC1 Variance      : "
    f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
)

print(
    f"PC2 Variance      : "
    f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
)

print(
    f"Total Variance    : "
    f"{total_variance * 100:.2f}%"
)

print("\nPCA experiment completed successfully!")