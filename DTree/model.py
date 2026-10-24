import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_excel("DecisionTreeDataset.xlsx")
print("Dataset:")
print(df)
df = df.drop("ID", axis=1)
# Outlook
df["Outlook"] = df["Outlook"].map({
    "Sunny": 0,
    "Overcast": 1,
    "Rain": 2
})
# Temperature
df["Temperature"] = df["Temperature"].map({
    "Hot": 0,
    "Mild": 1,
    "Cool": 2
})
# Humidity
df["Humidity"] = df["Humidity"].map({
    "High": 0,
    "Normal": 1
})

# Windy
df["Windy"] = df["Windy"].map({
    False: 0,
    True: 1
})
# Play_Tennis
df["Play_Tennis"] = df["Play_Tennis"].map({
    "No": 0,
    "Yes": 1
})

print("\nEncoded Dataset:")
print(df)



X = df[["Outlook", "Temperature", "Humidity", "Windy"]]

y = df["Play_Tennis"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nPredictions:")
print(y_pred)

print("\nActual Values:")
print(y_test.values)

print("\nAccuracy:", accuracy * 100, "%")


#,"Outlook", "Temperature", "Humidity", "Windy"
new_data = [[0, 2, 1, 0]]

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("\nNew Data Prediction: Yes - Play Tennis")
else:
    print("\nNew Data Prediction: No - Don't Play Tennis")


plt.figure(figsize=(15, 8))

plot_tree(
    model,
    feature_names=[
        "Outlook",
        "Temperature",
        "Humidity",
        "Windy"
    ],
    class_names=["No", "Yes"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Play Tennis")

plt.show()
