import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("knn_dataset68.csv")

print("Dataset:")
print(data)


X = data[["Height", "Weight", "Age"]]
y = data["Gender"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=5)


knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nActual values:")
print(y_test.values)

print("\nPredicted values:")
print(y_pred)

print("\nAccuracy:")
print(accuracy * 100, "%")


new_person = [[170, 68, 26]]

new_person_scaled = scaler.transform(new_person)

prediction = knn.predict(new_person_scaled)

if prediction[0] == 1:
    print("\nNew person prediction: Male")
else:
    print("\nNew person prediction: Female")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
"""
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))"""  
