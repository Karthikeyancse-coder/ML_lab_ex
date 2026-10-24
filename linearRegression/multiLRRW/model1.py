# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv("HousePricePrediction.csv")

print("First 5 Rows")
print(data.head())


X = data[['Area', 'Bedrooms', 'YearBuilt']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-------------------------")
print("Mean Squared Error :", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


print("\nModel Coefficients")
print("-------------------------")
print("Area Coefficient:", model.coef_[0])
print("Bedrooms Coefficient:", model.coef_[1])
print("Age Coefficient:", model.coef_[2])
print("Intercept:", model.intercept_)

new_house = pd.DataFrame({
    'Area':[2300],
    'Bedrooms':[4],
    'YearBuilt':[10]
})

predicted_price = model.predict(new_house)

print("\nPredicted Price = $", round(predicted_price[0],2))



plt.figure(figsize=(7,6))

plt.scatter(
    y_test,
    y_pred,
    color="green"
)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
    )

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Accuracy Graph (Actual vs Predicted)")
plt.grid(True)



plt.figure(figsize=(8,6))

plt.scatter(
    data["Area"],
    data["Price"],
    color="blue",
    label="Actual Data"
)

coef = np.polyfit(data["Area"], data["Price"], 1)

x = np.sort(data["Area"])
y_line = coef[0]*x + coef[1]

plt.plot(
    x,
    y_line,
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("Area (sq.ft)")
plt.ylabel("Price ($)")
plt.title("Area vs Price")
plt.legend()
plt.grid(True)

plt.show()
plt.show()
