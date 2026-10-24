# Import Libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


data = pd.read_csv("foodDelivery.csv")

X =data[['Distance_km']]

Y = data[['Delivery_Time_min']]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.50    , random_state=0
)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

r2 = r2_score(Y_test, Y_pred)


print("\nSlope:", model.coef_[0])
print("\nIntercept:", model.intercept_)
print("\nActual Values:", Y_test)
print("\nPredicted Values:", Y_pred)
print("\nR² Score:", r2)
print("\nAccuracy: {:.2f}%".format(r2 * 100))

plt.scatter(X_train, Y_train, color='blue', label='Training Data')
plt.scatter(X_test, Y_test, color='green', label='Test Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression Line')

plt.title("Simple Linear Regression")
plt.xlabel("Distance_km")
plt.ylabel("Delivery_Time_min")
plt.legend()

plt.show()