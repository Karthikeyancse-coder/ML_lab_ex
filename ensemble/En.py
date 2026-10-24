import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# ==========================================
# 1. LOAD DATASET
# ==========================================
file_path = "Titanic-Dataset.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print(f"Total Rows & Columns: {df.shape}\n")

# ==========================================
# 2. PREPROCESSING & FEATURE SELECTION
# ==========================================
selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[selected_features].copy()
y = df['Survived']

# Fill missing values
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())

# Map categorical text to numeric values (female: 0, male: 1)
X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# 4. INITIALIZE MODELS
# ==========================================
base_model = DecisionTreeClassifier(random_state=42)
rf_ensemble = RandomForestClassifier(n_estimators=100, random_state=42)
ada_ensemble = AdaBoostClassifier(n_estimators=100, random_state=42)

models = {
    "Single Decision Tree": base_model,
    "Random Forest (Bagging)": rf_ensemble,
    "AdaBoost (Boosting)": ada_ensemble
}

# ==========================================
# 5. TRAIN & EVALUATE
# ==========================================
accuracies = {}
cms = {}

print("=================== LAB EXPERIMENT RESULTS ===================")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    accuracies[name] = acc * 100
    cms[name] = cm
    
    print(f"\nModel: {name}")
    print(f"Accuracy Score: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

# ==========================================
# 6. VISUALIZATIONS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Chart 1: Accuracy Bar Comparison
colors = ['#e74c3c', '#2ecc71', '#3498db']
bars = axes[0].bar(accuracies.keys(), accuracies.values(), color=colors)
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Model Accuracy Comparison", fontweight="bold")
axes[0].set_ylim(60, 100)

for bar in bars:
    yval = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', fontweight='bold')

# Chart 2: Feature Importance via Random Forest
importances = pd.Series(rf_ensemble.feature_importances_, index=selected_features).sort_values()
importances.plot(kind='barh', ax=axes[1], color='#9b59b6')
axes[1].set_title("Feature Importance (Random Forest)", fontweight="bold")
axes[1].set_xlabel("Relative Importance Score")

plt.tight_layout()
plt.show()

# Chart 3: Confusion Matrices Grid
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, cm) in zip(axes, cms.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f"Confusion Matrix:\n{name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

plt.tight_layout()
plt.show()

# ==========================================
# 7. TEST SAMPLE PREDICTION (LIVE INFERENCE)
# ==========================================
print("\n=================== PASSENGER SURVIVAL PREDICTION ===================")

# Create custom test sample inputs
# Example Passenger 1: Female, 1st Class, 22 years old, paid high fare ($80)
passenger_1 = pd.DataFrame([{
    'Pclass': 1,
    'Sex': 0,        # 0 = female
    'Age': 22.0,
    'SibSp': 0,      # 0 siblings/spouses
    'Parch': 0,      # 0 parents/children
    'Fare': 80.0
}])

# Example Passenger 2: Male, 3rd Class, 30 years old, paid low fare ($7.25)
passenger_2 = pd.DataFrame([{
    'Pclass': 3,
    'Sex': 1,        # 1 = male
    'Age': 30.0,
    'SibSp': 1,
    'Parch': 0,
    'Fare': 7.25
}])

sample_passengers = {
    "Passenger 1 (1st Class Female, 22 yrs old)": passenger_1,
    "Passenger 2 (3rd Class Male, 30 yrs old)": passenger_2
}

for desc, sample_data in sample_passengers.items():
    print(f"\nEvaluating: {desc}")
    for name, model in models.items():
        prediction = model.predict(sample_data)[0]
        # Get probability score if supported
        prob = model.predict_proba(sample_data)[0][1] * 100
        
        status = "SURVIVED (Lived) 🟢" if prediction == 1 else "DID NOT SURVIVE (Died) 🔴"
        print(f"  ➜ [{name}]: {status} (Survival Chance: {prob:.1f}%)")

        