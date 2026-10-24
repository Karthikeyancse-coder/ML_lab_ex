 # Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv("house_price.csv")

print("Dataset:")
print(data.head())


X = data[['Area', 'Bedrooms', 'Age']]
y = data['Price']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)
print("Area Coefficient:", model.coef_[0])
print("Bedrooms Coefficient:", model.coef_[1])
print("Age Coefficient:", model.coef_[2])
print("Intercept:", model.intercept_)


new_house = [[2300, 4, 5]]
predicted_price = model.predict(new_house)

print("\nPredicted Price for New House: $", round(predicted_price[0], 2))


plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")

# Create 3D Figure
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot (Actual Data)
ax.scatter(
    data['Area'],
    data['Bedrooms'],
    data['Price'],
    color='blue',
    s=50,
    label='Actual Data'
)

# Create mesh grid for Area and Bedrooms
area_range = np.linspace(data['Area'].min(), data['Area'].max(), 20)
bedroom_range = np.linspace(data['Bedrooms'].min(), data['Bedrooms'].max(), 20)

Area_grid, Bedroom_grid = np.meshgrid(area_range, bedroom_range)

# Keep Age constant (average age)
Age_mean = data['Age'].mean()

# Predict price on the grid
Price_pred = model.predict(
    pd.DataFrame({
        'Area': Area_grid.ravel(),
        'Bedrooms': Bedroom_grid.ravel(),
        'Age': Age_mean
    })
)

Price_grid = Price_pred.reshape(Area_grid.shape)

# Plot regression plane
ax.plot_surface(
    Area_grid,
    Bedroom_grid,
    Price_grid,
    color='red',
    alpha=0.5
)

# Labels
ax.set_xlabel("Area (sq.ft)")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price ($)")
ax.set_title("3D Multiple Linear Regression: House Price Prediction")

plt.show()    

plt.show()
