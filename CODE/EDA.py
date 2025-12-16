import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load the dataset
file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

# =======================================================
# 1. Data Cleaning – Fix for ValueError
# =======================================================

# Drop the customerID column
df.drop('customerID', axis=1, inplace=True, errors='ignore')

# Handle TotalCharges: convert to numeric and drop NaNs
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Standardize SeniorCitizen data type
df['SeniorCitizen'] = df['SeniorCitizen'].replace({0: 'No', 1: 'Yes'}).astype('object')

print(f"Final number of rows after cleaning: {df.shape[0]}")
print("-" * 50)

# =======================================================
# 2. Exploratory Data Analysis (EDA) – Fix for FutureWarning
# =======================================================

numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# 2.1 Univariate Analysis: Summary statistics
print("### 2.1 Summary Statistics for Numerical Features")
print(df[numerical_cols].describe().T)
print("-" * 50)

# 2.2 Univariate Analysis: Distribution plots (Histograms)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Univariate Analysis: Distribution of Numerical Features', fontsize=16)

sns.histplot(df['tenure'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Tenure (Months)')

sns.histplot(df['MonthlyCharges'], kde=True, ax=axes[1], color='coral')
axes[1].set_title('Distribution of Monthly Charges')

sns.histplot(df['TotalCharges'], kde=True, ax=axes[2], color='lightgreen')
axes[2].set_title('Distribution of Total Charges')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('1_Numerical_Histograms.png')
plt.close()

# 2.3 Univariate Analysis: Bar charts (Count plots)
categorical_for_univariate = ['Churn', 'Contract', 'InternetService', 'PaymentMethod']
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
fig.suptitle('Univariate Analysis: Distribution of Key Categorical Features', fontsize=16)

for i, col in enumerate(categorical_for_univariate):
    # Fix for FutureWarning: add hue=col and legend=False
    sns.countplot(
        x=col,
        data=df,
        ax=axes[i],
        palette='Pastel2',
        hue=col,
        legend=False,
        order=df[col].value_counts().index
    )
    axes[i].set_title(f'Distribution of {col}', fontsize=14)
    axes[i].tick_params(axis='x', rotation=15)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('2_Categorical_BarCharts.png')
plt.close()

# 2.4 Bivariate Analysis: Contract vs Churn (Stacked Bar Chart)
contract_churn_pct = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
plt.figure(figsize=(8, 6))
contract_churn_pct.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='viridis')
plt.title('Churn Rate by Contract Type', fontsize=16)
plt.ylabel('Percentage of Customers', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Churn', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('3_Churn_by_Contract_pct.png')
plt.close()

# 2.5 Bivariate Analysis: MonthlyCharges vs Churn (Box Plot)
plt.figure(figsize=(8, 6))
# Fix for FutureWarning: add hue='Churn' and legend=False
sns.boxplot(
    x='Churn',
    y='MonthlyCharges',
    data=df,
    palette='cubehelix',
    hue='Churn',
    legend=False
)
plt.title('Distribution of Monthly Charges by Churn', fontsize=16)
plt.ylabel('Monthly Charges ($)', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('4_MonthlyCharges_by_Churn.png')
plt.close()

# 2.6 Bivariate Analysis: Tenure vs Churn (Box Plot)
plt.figure(figsize=(8, 6))
# Fix for FutureWarning: add hue='Churn' and legend=False
sns.boxplot(
    x='Churn',
    y='tenure',
    data=df,
    palette='Set2',
    hue='Churn',
    legend=False
)
plt.title('Distribution of Tenure by Churn', fontsize=16)
plt.ylabel('Tenure (Months)', fontsize=12)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig('Tenure_by_Churn_EN.png')
plt.close()

# 2.7 Multivariate Analysis: Correlation Heatmap
correlation_matrix = df[numerical_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=.5,
    linecolor='black'
)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.tight_layout()
plt.savefig('5_Correlation_Heatmap.png')
plt.close()

print("All required EDA plots have been completed and saved as image files.")
