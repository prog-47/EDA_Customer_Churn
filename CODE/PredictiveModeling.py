import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =======================================================
# 1. Loading, Cleaning, and Feature Engineering
# (to define the DataFrame df)
# =======================================================

file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(file_path)

# --- Data Cleaning ---
df.drop('customerID', axis=1, inplace=True, errors='ignore')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['SeniorCitizen'] = df['SeniorCitizen'].replace({0: 'No', 1: 'Yes'}).astype('object')

# --- Feature Engineering (Encoding) ---

# Label Encoding for binary features
binary_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'SeniorCitizen', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Churn'
]

for col in binary_cols:
    if col == 'gender':
        df[col] = df[col].replace({'Male': 1, 'Female': 0})
    else:
        df[col] = df[col].replace({
            'Yes': 1,
            'No': 0,
            'No phone service': 0,
            'No internet service': 0
        })

# One-Hot Encoding for multi-category features
multi_category_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=multi_category_cols, drop_first=True, dtype=int)

# =======================================================
# 2. Predictive Modeling
# =======================================================

print("Starting predictive model building...")

# --- A. Data Splitting and Feature Scaling ---
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numerical_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()

X_train[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
X_test[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])

# --- B. Training the Logistic Regression Model ---
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train, y_train)

# --- C. Prediction and Model Evaluation ---
y_pred = model.predict(X_test)

print("-" * 50)
print("### Logistic Regression Model Performance Evaluation:")

# Classification Report
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=['No Churn (0)', 'Churn (1)'],
    yticklabels=['No Churn (0)', 'Churn (1)']
)
plt.title('Confusion Matrix for Logistic Regression')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
