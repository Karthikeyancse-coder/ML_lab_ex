
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("accident.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

X = df.drop("Survived", axis=1)
y = df["Survived"]

numerical_features = [
    "Age",
    "Speed_of_Impact"
]

categorical_features = [
    "Gender",
    "Helmet_Used",
    "Seatbelt_Used"
]

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=None,
    min_samples_split=2
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n====================================")
print("Random Forest Classification Results")
print("====================================")

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy (%):")
print(f"{accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
feature_names = pipeline.named_steps[
    "preprocessing"
].get_feature_names_out()

from sklearn.tree import plot_tree
import numpy as np

# Get the trained Random Forest
rf_model = pipeline.named_steps["classifier"]

# Get the preprocessing object
preprocessor_fitted = pipeline.named_steps["preprocessing"]

# Transform the test data
X_test_transformed = preprocessor_fitted.transform(X_test)

# Select one sample to demonstrate the Random Forest
sample_index = 0

sample = X_test_transformed[sample_index:sample_index + 1]

actual_class = y_test.iloc[sample_index] if hasattr(y_test, "iloc") else y_test[sample_index]

# ==========================================================
# RANDOM FOREST - TREE VISUALIZATION
# ==========================================================

from sklearn.tree import plot_tree
import numpy as np

# Get the trained Random Forest
rf_model = pipeline.named_steps["classifier"]

# Get the preprocessing object
preprocessor_fitted = pipeline.named_steps["preprocessing"]

# Transform the test data
X_test_transformed = preprocessor_fitted.transform(X_test)

# Select one sample to demonstrate the Random Forest
sample_index = 0

sample = X_test_transformed[sample_index:sample_index + 1]

actual_class = y_test.iloc[sample_index] if hasattr(y_test, "iloc") else y_test[sample_index]

# ----------------------------------------------------------
# Get predictions from 3 trees
# ----------------------------------------------------------

tree_predictions = []

for i in range(3):

    prediction = rf_model.estimators_[i].predict(sample)[0]

    tree_predictions.append(prediction)


# ----------------------------------------------------------
# Majority voting
# ----------------------------------------------------------

final_prediction = max(
    set(tree_predictions),
    key=tree_predictions.count
)


# ----------------------------------------------------------
# Print results
# ----------------------------------------------------------
print("\n====================================")
print("Random Forest Tree Voting")
print("====================================")

print("\nActual Class:", actual_class)

for i, prediction in enumerate(tree_predictions):
    print(f"Tree-{i + 1} Prediction:", prediction)

print("\nMajority Voting Result:", final_prediction)


# ==========================================================
# GRAPH 1: RANDOM FOREST ARCHITECTURE
# ==========================================================

fig = plt.figure(figsize=(18, 12))

# Main title
fig.suptitle(
    "Random Forest Classification - Tree",
    fontsize=22,
    fontweight="bold"
)

# ----------------------------------------------------------
# Tree 1
# ----------------------------------------------------------

ax1 = fig.add_subplot(2, 3, 1)

plot_tree(
    rf_model.estimators_[0],
    feature_names=feature_names,
    class_names=[str(c) for c in rf_model.classes_],
    filled=True,
    rounded=True,
    max_depth=2,
    fontsize=7,
    ax=ax1
)

ax1.set_title(
    f"Tree-1\nPrediction: {tree_predictions[0]}",
    fontsize=13,
    fontweight="bold"
)


# ----------------------------------------------------------
# Tree 2
# ----------------------------------------------------------

ax2 = fig.add_subplot(2, 3, 2)

plot_tree(
    rf_model.estimators_[1],
    feature_names=feature_names,
    class_names=[str(c) for c in rf_model.classes_],
    filled=True,
    rounded=True,
    max_depth=2,
    fontsize=7,
    ax=ax2
)

ax2.set_title(
    f"Tree-2\nPrediction: {tree_predictions[1]}",
    fontsize=13,
    fontweight="bold"
)


# ----------------------------------------------------------
# Tree 3
# ----------------------------------------------------------

ax3 = fig.add_subplot(2, 3, 3)

plot_tree(
    rf_model.estimators_[2],
    feature_names=feature_names,
    class_names=[str(c) for c in rf_model.classes_],
    filled=True,
    rounded=True,
    max_depth=2,
    fontsize=7,
    ax=ax3
)

ax3.set_title(
    f"Tree-3\nPrediction: {tree_predictions[2]}",
    fontsize=13,
    fontweight="bold"
)


# ----------------------------------------------------------
# Input section
# ----------------------------------------------------------

ax4 = fig.add_subplot(2, 3, 4)

ax4.axis("off")

ax4.text(
    0.5,
    0.5,
    "INPUT INSTANCE\n\nAccident Data",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    bbox=dict(
        boxstyle="round,pad=0.8",
        facecolor="lightblue",
        edgecolor="black"
    )
)


# ----------------------------------------------------------
# Tree predictions
# ----------------------------------------------------------

ax5 = fig.add_subplot(2, 3, 5)

ax5.axis("off")

prediction_text = (
    f"Tree-1  →  {tree_predictions[0]}\n\n"
    f"Tree-2  →  {tree_predictions[1]}\n\n"
    f"Tree-3  →  {tree_predictions[2]}"
)

ax5.text(
    0.5,
    0.5,
    prediction_text,
    ha="center",
    va="center",
    fontsize=17,
    bbox=dict(
        boxstyle="round,pad=0.8",
        facecolor="lightyellow",
        edgecolor="black"
    )
)


# ----------------------------------------------------------
# Majority voting / final class
# ----------------------------------------------------------

ax6 = fig.add_subplot(2, 3, 6)

ax6.axis("off")

ax6.text(
    0.5,
    0.65,
    "MAJORITY VOTING",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold"
)

ax6.text(
    0.5,
    0.35,
    f"FINAL CLASS\n\n{final_prediction}",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
    color="darkgreen",
    bbox=dict(
        boxstyle="round,pad=0.8",
        facecolor="lightgreen",
        edgecolor="black"
    )
)


plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.show()