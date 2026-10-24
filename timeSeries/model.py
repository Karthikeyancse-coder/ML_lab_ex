import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


data = pd.read_csv("retail_sales.csv")

print("Column names in CSV:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())

date_column = data.columns[0]
sales_column = data.columns[1]

print("\nDate column:", date_column)
print("Sales column:", sales_column)



data[date_column] = pd.to_datetime(data[date_column])

data.set_index(date_column, inplace=True)

# Rename sales column
data.rename(columns={sales_column: "Sales"}, inplace=True)

# Convert sales values to numbers
data["Sales"] = pd.to_numeric(data["Sales"], errors="coerce")

# Remove missing values
data.dropna(inplace=True)


print("\nCleaned data:")
print(data.head())

print("\nNumber of records:", len(data))

plt.figure(figsize=(12, 5))

plt.plot(data["Sales"], color="blue")

plt.title("Monthly Retail Sales")
plt.xlabel("Year")
plt.ylabel("Sales")

plt.grid(True)

plt.show()

data["Moving_Average"] = data["Sales"].rolling(window=12).mean()

plt.figure(figsize=(12, 5))

plt.plot(
    data["Sales"],
    label="Actual Sales",
    color="blue"
)

plt.plot(
    data["Moving_Average"],
    label="12-Month Moving Average",
    color="red",
    linewidth=2
)

plt.title("Retail Sales Trend")
plt.xlabel("Year")
plt.ylabel("Sales")

plt.legend()
plt.grid(True)

plt.show()


decomposition = seasonal_decompose(
    data["Sales"],
    model="additive",
    period=12
)

decomposition.plot()

plt.suptitle(
    "Retail Sales - Trend and Seasonality",
    fontsize=14
)

plt.show()
