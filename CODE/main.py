import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options for easier reading of large tables
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# =======================================================
# 1. Data Loading
# =======================================================

# Load the uploaded CSV file
file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

# Display the first 5 rows to verify successful loading
print("### 1.1 First 5 Rows of the Dataset (Head of Dataset)")
print(df.head())
print("-" * 50)

# =======================================================
# 2. Initial Data Inspection
# =======================================================

# Check the number of rows, columns, data types (dtypes),
# and non-null values (DataFrame Info)
print("### 2.1 Dataset Information (DataFrame Info)")
df.info()
print("-" * 50)

# Check dataset dimensions (Shape)
print(f"### 2.2 Dataset Dimensions: {df.shape[0]} rows and {df.shape[1]} columns.")
print("-" * 50)
