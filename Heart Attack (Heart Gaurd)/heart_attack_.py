# Generated from: heart_attack_.ipynb
# Converted at: 2025-12-22T16:44:17.129Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report,precision_score,recall_score,ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.ensemble import RandomForestClassifier



df = pd.read_csv('Heart_Disease_Prediction.csv')

df.shape

df.head(30)

df.info()

df.describe()

df.isnull().sum()

df["Number of vessels fluro"].value_counts()

df["Slope of ST"].value_counts()

df["Thallium"].value_counts()

df["FBS over 120"].value_counts()

df["Slope of ST"].value_counts()

prediciton = {
    "Presence": 1,
    "Absence": 0 
}
df["Heart Disease"] = df["Heart Disease"].map(prediciton)


df.head()

scaler = StandardScaler()
ohe = OneHotEncoder()
le = LabelEncoder()

num_cols = ["Age","BP","Cholesterol","Max HR","Thallium","ST depression"]

print(df.columns)

df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()

binary_cols = [
    'Sex',
    'FBS over 120',
    'Exercise angina',
    'Heart Disease'
]

for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df.head()

df = pd.get_dummies(
    df,
    columns=[
        'Chest pain type',
        'EKG results',
        'Slope of ST',
        'Number of vessels fluro'
    ],
    drop_first=True
)


df.head()

df.shape

df.columns

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'FBS over 120', 'Max HR',
       'Exercise angina', 'ST depression', 'Thallium', 
       'Chest pain type_2', 'Chest pain type_3', 'Chest pain type_4',
       'EKG results_1', 'EKG results_2', 'Slope of ST_2', 'Slope of ST_3',
       'Number of vessels fluro_1', 'Number of vessels fluro_2',
       'Number of vessels fluro_3']]

y = df['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

log_model = LogisticRegression()

lg_model = log_model.fit(X_train,y_train)
y_log_pred = lg_model.predict(X_test)
                         

print("Logistic Regression Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_log_pred))
print("Classification Report:")
print(classification_report(y_test, y_log_pred))

cm = confusion_matrix(y_test, y_log_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Absence", "Presence"]
)

disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)

print("Random Forest Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_rf_pred))
print("Classification Report:")
print(classification_report(y_test, y_rf_pred))

cm = confusion_matrix(y_test, y_rf_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Absence", "Presence"]
)

disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()

# ## Test case #1
# ### Raw Patient Profile:
# 
# ```Age: 65 (Older)
# Sex: Male (1)
# BP: 160 (High)
# Cholesterol: 280 (High)
# FBS over 120: Yes (1)
# Max HR: 120 (Low, indicating poor fitness)
# Exercise Angina: Yes (1)
# ST Depression: 2.5 (Significant)
# Thallium Scan: Reversible Defect (3)
# Chest Pain Type: Type 4 (Asymptomatic)
# EKG Results: ST-T abnormality (2)
# Slope of ST: Flat (2)
# Number of Vessels: 3 (Major blockage)


high_risk_test_case = {
    'Age': 1.7,  # Scaled high value
    'Sex': 1.0,
    'BP': 1.5,   # Scaled high value
    'Cholesterol': 2.1,  # Scaled high value
    'FBS over 120': 1.0,
    'Max HR': -1.2, # Scaled low value
    'Exercise angina': 1.0,
    'ST depression': 1.8, # Scaled high value
    'Thallium': -0.8, # Scaled value for '3'
    # One-Hot Encoded Features
    'Chest pain type_2': 0, 'Chest pain type_3': 0, 'Chest pain type_4': 1,
    'EKG results_1': 0, 'EKG results_2': 1,
    'Slope of ST_2': 1, 'Slope of ST_3': 0,
    'Number of vessels fluro_1': 0, 'Number of vessels fluro_2': 0, 'Number of vessels fluro_3': 1
}

# Create a DataFrame from the dictionary
high_risk_df = pd.DataFrame([high_risk_test_case])

# Make sure the columns are in the exact same order as your training data (X)
high_risk_df = high_risk_df[X.columns]

# --- Now you can make a prediction ---
# prediction = model.predict(high_risk_df)
# print("High-Risk Prediction:", prediction)

print("--- High-Risk Test Case (Ready for Model) ---")
print(high_risk_df)

test_case = log_model.predict(high_risk_df)[0]
print("High-Risk Prediction (Logistic Regression):", test_case)

test_case = rf_model.predict(high_risk_df)[0]
print("High-Risk Prediction (Random Forest):", test_case)

# ## Test case #2
# ### Raw Patient Profile:
# 
# ``` Age: 35 (Younger)
# Sex: Female (0)
# BP: 120 (Normal)
# Cholesterol: 180 (Normal)
# FBS over 120: No (0)
# Max HR: 180 (High, indicating good fitness)
# Exercise Angina: No (0)
# ST Depression: 0.5 (Minimal)
# Thallium Scan: Normal (7)
# Chest Pain Type: Type 2 (Non-anginal pain)
# EKG Results: Normal (0)
# Slope of ST: Upsloping (1)
# Number of Vessels: 0 (No blockage) ```


low_risk_test_case = {
    'Age': -1.2, # Scaled low value
    'Sex': 0.0,
    'BP': -0.5,  # Scaled low value
    'Cholesterol': -0.3, # Scaled low value
    'FBS over 120': 0.0,
    'Max HR': 1.9,  # Scaled high value
    'Exercise angina': 0.0,
    'ST depression': -0.9, # Scaled low value
    'Thallium': 1.1, # Scaled value for '7'
    # One-Hot Encoded Features
    'Chest pain type_2': 1, 'Chest pain type_3': 0, 'Chest pain type_4': 0,
    'EKG results_1': 0, 'EKG results_2': 0,
    'Slope of ST_2': 0, 'Slope of ST_3': 0,
    'Number of vessels fluro_1': 0, 'Number of vessels fluro_2': 0, 'Number of vessels fluro_3': 0
}

# Create a DataFrame from the dictionary
low_risk_df = pd.DataFrame([low_risk_test_case])

# Make sure the columns are in the exact same order as your training data (X)
low_risk_df = low_risk_df[X.columns]

print("\n--- Low-Risk Test Case (Ready for Model) ---")
print(low_risk_df)

test_pred = log_model.predict(low_risk_df)[0]
print("Low-Risk Prediction (Logistic Regression):", test_pred)

test_case = rf_model.predict(low_risk_df)[0]
print("Low-Risk Prediction (Random Forest):", test_case)

import joblib
import pandas as pd

# Assume 'model' is your trained RandomForestClassifier
# and 'scaler' is your fitted StandardScaler

# 1. Save the trained model
joblib.dump(model, 'heart_disease_model.pkl')

# 2. Save the fitted scaler
joblib.dump(scaler, 'scaler.pkl')

# 3. Save the full column list from your processed DataFrame (X)
# This is crucial for making sure the test case has the right columns
joblib.dump(list(X.columns), 'model_columns.pkl')

print("Model, scaler, and columns saved successfully!")