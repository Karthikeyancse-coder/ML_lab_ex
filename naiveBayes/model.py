# Naive Bayes Classification - Bank Personal Loan Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("Bank_Personal_Loan.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# Personal Loan is the variable we want to predict
X = df.drop("Personal Loan", axis=1)

y = df["Personal Loan"]


X = X.drop(["ID", "ZIP Code"], axis=1)

print("\nFeatures used for prediction:")
print(X.columns)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


model = GaussianNB()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")

#

new_customer1 = [[
    35,     # Age
    10,     # Experience
    60,     # Income
    2,      # Family
    2.0,    # CCAvg
    1,      # Education
    0,      # Mortgage
    0,      # Securities Account
    0,      # CD Account
    1,      # Online
    1       # CreditCard
]]

prediction = model.predict(new_customer1)
probability = model.predict_proba(new_customer1)

print("\nNew Customer Prediction:")

if prediction[0] == 1:
    print("YES - Customer gets the loan.")
else:
    print("NO - Customer does not get the loan.")

print("\nPrediction probabilities:")
print("No Loan:",   round(probability[0][0] * 100, 2), "%")
print("Personal Loan:",  round(probability[0][1] * 100, 2), "%")

new_customer2 = [[
    45,    # Age
    20,    # Experience
    100,   # Income
    4,     # Family
    3.5,   # CCAvg
    3,     # Education
    150,   # Mortgage
    0,     # Securities Account
    1,     # CD Account
    1,     # Online
    1      # CreditCard
]]

prediction1 = model.predict(new_customer2)
probability1 = model.predict_proba(new_customer2)

if prediction1[0] == 1:
    print("YES - Customer gets the loan")
else:
    print("NO - Customer does not get the loan")


print("\nPrediction probabilities:")
print("No Loan:",      round(probability1[0][0] * 100, 2), "%")
print("Personal Loan:", round(probability1[0][1] * 100, 2), "%")






















"""
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""