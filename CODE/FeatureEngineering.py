import pandas as pd
import numpy as np

# Read CSV file
file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

# --- 1. Data Cleaning and Preparation ---
df.drop('customerID', axis=1, inplace=True, errors='ignore')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['SeniorCitizen'] = df['SeniorCitizen'].replace({0: 'No', 1: 'Yes'}).astype('object')

print("Initial data cleaning completed.")
print("-" * 50)

# --- 2. Label Encoding for Binary Features ---

# Identify binary columns (Yes/No) including the target variable (Churn)
binary_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'SeniorCitizen', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Churn'
]

# Apply encoding: Male/Yes/No_phone_service = 1, Female/No = 0
for col in binary_cols:
    if col == 'gender':
        # Special case for gender
        df[col] = df[col].replace({'Male': 1, 'Female': 0})
    else:
        # General case for other binary variables (Yes/No)
        df[col] = df[col].replace({
            'Yes': 1,
            'No': 0,
            'No phone service': 0,
            'No internet service': 0
        })

print("Label Encoding applied to binary variables.")
print("-" * 50)

# --- 3. One-Hot Encoding for Multi-Category Features ---

# Identify remaining multi-category columns
multi_category_cols = df.select_dtypes(include='object').columns.tolist()

print(f"Remaining columns for One-Hot Encoding: {multi_category_cols}")

# Apply One-Hot Encoding using get_dummies
df = pd.get_dummies(
    df,
    columns=multi_category_cols,
    drop_first=True,
    dtype=int
)

print("One-Hot Encoding applied successfully.")
print("-" * 50)

# --- 4. Display Final Result ---
print("### DataFrame after Feature Engineering:")
print(f"New shape of the dataset: {df.shape}")
print("Final column data types:")
print(df.dtypes.value_counts())
print("\nFirst 5 rows after encoding:")
print(df.head())
