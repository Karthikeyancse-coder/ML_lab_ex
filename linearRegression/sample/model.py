# Import Libraries
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Dataset
X = [[2], [3], [4], [5], [6], [7], [8], [9]]
Y = [50, 60, 65, 70, 80, 85, 90, 95]

# Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0
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
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Actual Values:", Y_test)
print("Predicted Values:", Y_pred)
print("R² Score:", r2)
print("Accuracy: {:.2f}%".format(r2 * 100))

# Graph
plt.scatter(X_train, Y_train, color='blue', label='Training Data')
plt.scatter(X_test, Y_test, color='green', label='Test Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')

plt.title("Simple Linear Regression")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()

plt.show()