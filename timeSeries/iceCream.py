import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv("icecream.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date")

# Set Date as index
df.set_index("Date", inplace=True)

# Sales data
sales = df["Sales_Units"]



decomposition = seasonal_decompose(
    sales,
    model="additive",
    period=12
)



plt.figure(figsize=(10, 5))

plt.plot(
    decomposition.trend,
    color="blue",
    linewidth=2
)

plt.title("Ice Cream Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales Units")
plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    decomposition.seasonal,
    color="orange",
    linewidth=2
)

plt.title("Ice Cream Sales Seasonal Pattern")
plt.xlabel("Date")
plt.ylabel("Seasonal Effect")
plt.grid(True)

plt.tight_layout()
plt.show()



season_sales = df.groupby("Season")["Sales_Units"].mean()

print("\nAverage Sales by Season:")
print(season_sales)

best_season = season_sales.idxmax()

print("\nBest Historical Season:", best_season)
print(
    "Average Sales:",
    round(season_sales.max(), 2),
    "units"
)



model = ExponentialSmoothing(
    sales,
    trend="add",
    seasonal="add",
    seasonal_periods=12
)

fit = model.fit()

# Forecast next 12 months
forecast = fit.forecast(12)


print("\nNext 12 Months Forecast:")

for date, value in forecast.items():
    print(
        date.strftime("%Y-%m"),
        "->",
        round(value, 2),
        "units"
    )



best_forecast_month = forecast.idxmax()
best_forecast_sales = forecast.max()

print("\nBest Forecast Month:",
      best_forecast_month.strftime("%B %Y"))

print(
    "Forecasted Sales:",
    round(best_forecast_sales, 2),
    "units"
)