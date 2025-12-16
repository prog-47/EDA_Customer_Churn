import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

# Note: It is assumed that df is the DataFrame loaded at the beginning

# 1. Check for duplicate rows based on the 'customerID' column
# df.duplicated() returns True for duplicated rows
# subset=['customerID'] means duplication is checked only on 'customerID'
# .sum() counts the number of True values (i.e., duplicated rows)

duplicate_ids_count = df.duplicated(subset=['customerID']).sum()

print("### Checking duplicate customer IDs (customerID):")
print(f"Number of rows with duplicated customerID: {duplicate_ids_count}")
print("-" * 50)

# 2. Remove duplicate rows based on 'customerID'
if duplicate_ids_count > 0:
    # df.drop_duplicates removes duplicated rows, keeping the first occurrence (keep='first')
    # subset=['customerID'] ensures duplicates are identified based only on this column
    # inplace=True applies the change directly to the DataFrame
    df.drop_duplicates(subset=['customerID'], keep='first', inplace=True)

    # Final verification after removal
    print("Duplicate rows based on 'customerID' have been successfully removed.")
    print(f"New number of rows in the dataset: {df.shape[0]}")
else:
    print("No duplicate customer IDs (customerID) were found. The data is clean.")

print("-" * 50)

# --- 3.1 Handling the 'TotalCharges' column ---

# First, convert the 'TotalCharges' column to a numerical type
# Using errors='coerce' will convert any non-numeric values
# (including blank spaces or empty values) into NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for missing values (NaN) after conversion
missing_values = df.isnull().sum()
print("### 3.2 Number of missing values in each column:")
print(missing_values[missing_values > 0])
print("-" * 50)

# We observe 11 missing values in 'TotalCharges'
# Since this number is very small (less than 0.2% of 7043 rows),
# we will drop these rows to avoid introducing bias
df.dropna(inplace=True)

# Verification after dropping missing values
print(f"### 3.3 Number of rows after removing missing values: {df.shape[0]}")
print("-" * 50)

# --- 3.4 Drop unnecessary column (customerID) ---
df.drop('customerID', axis=1, inplace=True, errors='ignore')
print("The 'customerID' column has been removed as it is not useful for analysis.")

print("-" * 50)

# Split columns into Categorical and Numerical
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("### 4.1 Categorical columns (qualitative variables):")
print(categorical_cols)

print("\n### 4.2 Numerical columns (quantitative variables):")
print(numerical_cols)
