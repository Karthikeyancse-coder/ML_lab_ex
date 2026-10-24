
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
Dataset = pd.read_csv("car.csv")

print("Dataset Shape:")
print(Dataset.shape)

print("\nFirst 5 Rows:")
print(Dataset.head())

print("\nColumn Names:")
print(Dataset.columns)


X = Dataset.iloc[:, :-1]
y = Dataset.iloc[:, -1]

print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget Classes:")
print(y.unique())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_encoded = encoder.fit_transform(X_train)

X_test_encoded = encoder.transform(X_test)


print("\nEncoded Training Data Shape:")
print(X_train_encoded.shape)

print("\nEncoded Testing Data Shape:")
print(X_test_encoded.shape)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded)

X_test_scaled = scaler.transform(X_test_encoded)

classifier = SVC(
    C=10.0,
    kernel="linear",
    decision_function_shape="ovr"
)

classifier.fit(
    X_train_scaled,
    y_train
)

y_train_pred = classifier.predict(X_train_scaled)

y_test_pred = classifier.predict(X_test_scaled)


train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

train_f1 = f1_score(
    y_train,
    y_train_pred,
    average="weighted"
)
 

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted"
)



print("\n")
print("=" * 50)
print("SVM CLASSIFICATION RESULTS")
print("=" * 50)

print("\nTraining Results")
print("-" * 30)

print("Training Accuracy:",
      round(train_accuracy, 4))

print("Training F1 Score:",
      round(train_f1, 4))


print("\nTesting Results")
print("-" * 30)

print("Testing Accuracy:",
      round(test_accuracy, 4))

print("Testing F1 Score:",
      round(test_f1, 4))


pca = PCA(n_components=2)

X_all_encoded = encoder.transform(X)

X_all_scaled = scaler.transform(X_all_encoded)

X_pca = pca.fit_transform(X_all_scaled)



print("\n")
print("=" * 50)
print("PCA INFORMATION")
print("=" * 50)

print("\nExplained Variance Ratio:")

print(
    "PC1:",
    round(pca.explained_variance_ratio_[0], 4)
)

print(
    "PC2:",
    round(pca.explained_variance_ratio_[1], 4)
)

print(
    "\nTotal Variance Explained:",
    round(
        pca.explained_variance_ratio_.sum(),
        4
    )
)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = (
    train_test_split(
        X_pca,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
)

visual_encoder = LabelEncoder()

y_all_encoded = visual_encoder.fit_transform(y)

y_train_pca_encoded = visual_encoder.transform(
    y_train_pca
)

y_test_pca_encoded = visual_encoder.transform(
    y_test_pca
)

class_names = visual_encoder.classes_

print("\nClasses used for visualization:")
print(class_names)

x_min = X_pca[:, 0].min() - 0.8
x_max = X_pca[:, 0].max() + 0.8

y_min = X_pca[:, 1].min() - 0.8
y_max = X_pca[:, 1].max() + 0.8

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 600),
    np.linspace(y_min, y_max, 600)
)

grid_points = np.c_[
    xx.ravel(),
    yy.ravel()
]

svm_visual = SVC(
    C=10.0,
    kernel="linear",
    decision_function_shape="ovr"
)

svm_visual.fit(
    X_train_pca,
    y_train_pca_encoded
)


Z = svm_visual.predict(grid_points)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(14, 9))


plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.15,
    cmap="coolwarm"
)


plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train_pca_encoded,
    cmap="coolwarm",
    marker="o",
    s=55,
    edgecolors="black",
    linewidths=0.5,
    alpha=0.75,
    label="Training Data"
)


plt.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=y_test_pca_encoded,
    cmap="coolwarm",
    marker="^",
    s=90,
    edgecolors="black",
    linewidths=0.8,
    alpha=0.9,
    label="Test Data"
)


plt.scatter(
    svm_visual.support_vectors_[:, 0],
    svm_visual.support_vectors_[:, 1],
    s=200,
    facecolors="none",
    edgecolors="black",
    linewidths=2.5,
    label="Support Vectors"
)


colors = plt.cm.tab10(
    np.linspace(0, 1, len(class_names))
)

for class_index, class_name in enumerate(class_names):

    y_binary = np.where(
        y_train_pca_encoded == class_index,
        1,
        -1
    )

    binary_svm = SVC(
        C=10.0,
        kernel="linear"
    )

    binary_svm.fit(
        X_train_pca,
        y_binary
    )

    # Calculate decision function
    decision_values = binary_svm.decision_function(
        grid_points
    )

    decision_values = decision_values.reshape(
        xx.shape
    )

    
    plt.contour(
        xx,
        yy,
        decision_values,
        levels=[0],
        colors=[colors[class_index]],
        linewidths=3
    )

    plt.contour(
        xx,
        yy,
        decision_values,
        levels=[1],
        colors=[colors[class_index]],
        linestyles="--",
        linewidths=1.5,
        alpha=0.7
    )

    plt.contour(
        xx,
        yy,
        decision_values,
        levels=[-1],
        colors=[colors[class_index]],
        linestyles="--",
        linewidths=1.5,
        alpha=0.7
    )


handles, _ = plt.gca().get_legend_handles_labels()

class_handles = []

for i, class_name in enumerate(class_names):

    class_handle = plt.scatter(
        [],
        [],
        c=[colors[i]],
        s=80,
        edgecolors="black",
        label=class_name
    )

    class_handles.append(class_handle)

legend1 = plt.legend(
    handles=class_handles,
    title="Classes",
    loc="upper right"
)

plt.gca().add_artist(legend1)

plt.legend(
    loc="best",
    title="SVM Visualization"
)

plt.xlabel(
    "Principal Component 1",
    fontsize=13
)

plt.ylabel(
    "Principal Component 2",
    fontsize=13
)

plt.title(
    "SVM Decision Hyperplanes with Support Vectors",
    fontsize=16,
    fontweight="bold"
)

plt.grid(
    alpha=0.25,
    linestyle="--"
)
plt.tight_layout()

plt.show()