# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Load Dataset
df = pd.read_excel("Credit_Risk_Prediction_1.xlsx")

# Display Dataset
print("Original Dataset")
print(df)

# Dataset Information
print("\nDataset Information")
print(df.info())

# Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Fill Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Convert Data Type
df["Age"] = df["Age"].astype(int)

# Mean and Standard Deviation
print("\nMean")
print(np.mean(df[["Age","Credit_Score","Existing_Debt"]]))

print("\nStandard Deviation")
print(np.std(df[["Age","Credit_Score","Existing_Debt"]]))



# -------------------------------
# Standard Scaling
# -------------------------------
df_standard = df.copy()

standard = StandardScaler()

df_standard[["Age","Credit_Score","Existing_Debt","Income","Loan_Amount","Loan_Term_Months"]] = standard.fit_transform(
    df_standard[["Age","Credit_Score","Existing_Debt","Income","Loan_Amount","Loan_Term_Months"]]
)

print("\nDataset after StandardScaler")
print(df_standard)

# -------------------------------
# Min-Max Scaling
# -------------------------------
df_minmax = df.copy()

minmax = MinMaxScaler()

df_minmax[["Age","Credit_Score","Existing_Debt","Income","Loan_Amount","Loan_Term_Months"]] = minmax.fit_transform(
    df_minmax[["Age","Credit_Score","Existing_Debt","Income","Loan_Amount","Loan_Term_Months"]]
)

print("\nDataset after MinMaxScaler")
print(df_minmax)

# Save Processed Datasets
df_standard.to_excel("standard_scaled.xlsx", index=False)
df_minmax.to_excel("minmax_scaled.xlsx", index=False)

print("\nDatasets Saved Successfully")
