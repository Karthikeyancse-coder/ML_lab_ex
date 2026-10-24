# Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Dataset

data = pd.read_csv("Food_Delivery_Times.csv")

X =data[['Distance_km']]

Y = data[['Delivery_Time_min']]

# Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.40, random_state=0
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, Y_train)

# Predict Test Data
Y_pred = model.predict(X_test)

# Calculate R² Score
r2 = r2_score(Y_test, Y_pred)

# Display Results
print("\nSlope:", model.coef_[0])
print("\nIntercept:", model.intercept_)
print("\nActual Values:", Y_test)
print("\nPredicted Values:", Y_pred)
print("\nR² Score:", r2)
print("\nAccuracy: {:.2f}%".format(r2 * 100))

# Graph
plt.scatter(X_train, Y_train, color='blue', label='Training Data')
plt.scatter(X_test, Y_test, color='green', label='Test Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')

plt.title("Simple Linear Regression")
plt.xlabel("Distance_km")
plt.ylabel("Delivery_Time_min")
plt.legend()

plt.show()

"""
# Scatter Plot: Actual vs Predicted
plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, color='blue')

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=2
)

plt.title("Actual Price vs Predicted Price")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.grid(True)

plt.show()
"""