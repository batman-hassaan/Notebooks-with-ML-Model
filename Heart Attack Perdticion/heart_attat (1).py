# Generated from: heart_attat.ipynb
# Converted at: 2025-12-22T08:48:05.309Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score,precision_score, recall_score,r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression


file_path = ("heart_attack_vaccine_data.csv")
df = pd.read_csv(file_path);

print(df.info())

df.describe()

df.info()

df = df.drop_duplicates()


df.head(4)

df.replace({'Yes': 1, 'No': 0}, inplace=True)

df.columns
print(df.shape)

df.head(4)

# Create a flag column
df['Had_Heart_Attack'] = df['Heart Attack Date'].notna().astype(int)

# Replace severity and outcome if NaN
df['Severity'] = df['Severity'].fillna('None')
df['Outcome'] = df['Outcome'].fillna('None')

# For Pre-existing Conditions
df['Pre-existing Conditions'] = df['Pre-existing Conditions'].fillna('None')
df['Heart Attack Date'] = pd.to_datetime(df['Heart Attack Date'], errors='coerce')



df.head(4)


print(df.shape)

bp_mapping = {
    'Normal': 1,
    'Elevated': 2,
    'High': 3,
    'Very High': 4
}
df['BP_Score'] = df['Blood Pressure'].map(bp_mapping)


dose_mapping = {
    '1st Dose': 1,
    '2nd Dose': 2,
    'Booster': 3
}
df['Vaccine Dose'] = df['Vaccine Dose'].map(dose_mapping)

gender_mapping = {
    'Male': 1,
    'Female': 0
}
df['Gender'] = df['Gender'].map(gender_mapping)

df['BP_Score'] = df['BP_Score'].fillna(0)
df['BP_Score'] = df['BP_Score'].astype(int)

df['BP_Score']

df.head(4)

# Convert Vaccination Date to datetime
df['Vaccination Date'] = pd.to_datetime(df['Vaccination Date'], errors='coerce')

# Now create the Time_To_Heart_Attack column
df['Time_To_Heart_Attack'] = (df['Heart Attack Date'] - df['Vaccination Date']).dt.days


df['Days_Since_Vaccination'] = (
    pd.Timestamp.today() - df['Vaccination Date']
).dt.days


df['Had_Heart_Attack'] = df['Heart Attack Date'].notna().astype(int)


df['Vaccination Date'] = pd.to_datetime(df['Vaccination Date'], errors='coerce')

df['Vax_Year'] = df['Vaccination Date'].dt.year
df['Vax_Month'] = df['Vaccination Date'].dt.month
df['Vax_Day'] = df['Vaccination Date'].dt.day


df.head()

df.columns

df["Time_To_Heart_Attack"].head(20)

ohe = OneHotEncoder(sparse_output=False, drop='first')

df["Pre-existing Conditions"].value_counts()

encoded_array = ohe.fit_transform(df[["Pre-existing Conditions"]])

encoded_cols = ohe.get_feature_names_out(["Pre-existing Conditions"])


encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)


df = pd.concat([df, encoded_df], axis=1)


df.drop("Pre-existing Conditions", axis=1, inplace=True)


df.head()

df["Outcome"].value_counts()

df_outcome_ohe = pd.get_dummies(
    df["Outcome"],
    prefix="Outcome",
    dtype=int
)

df = pd.concat([df, df_outcome_ohe], axis=1)


df_location_ohe = pd.get_dummies(
    df["Location"],       # Location column
    prefix="Location",    # Prefix for new columns
    dtype=int
)

# Original dataframe ke saath concat kar do
df = pd.concat([df, df_location_ohe], axis=1)


df.drop("Outcome", axis=1, inplace=True)

df["Severity"].value_counts()



severity_map = {
    "None": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Fatal": 4
}

df["Severity"] = df["Severity"].map(severity_map)

df["Time_To_Heart_Attack"].value_counts()

df.head()

df.to_csv('clean_vaccine_heart_data.csv', index=False)


df.dtypes

# Recompute the numeric-only dataframe
numeric_df = df.select_dtypes(include=['int64', 'float64', 'int32'])

# Compute and plot correlation
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='bwr',center=0)
plt.title('Correlation Heatmap (Blue = Negative, Red = Positive)', fontsize=14)
plt.show()





heart_attacks_by_city = df.groupby('Location')['Had_Heart_Attack'].sum().sort_values(ascending=False)

# Set plot style for better visuals
plt.figure(figsize=(12, 6))
sns.barplot(x=heart_attacks_by_city.index, y=heart_attacks_by_city.values)

# Add titles and labels
plt.title('Number of Heart Attack Patients by City', fontsize=18, fontweight='bold', color='#4B0082')
plt.xlabel('City', fontsize=14)
plt.ylabel('Number of Heart Attack Patients', fontsize=14 )

# Improve x-axis labels with better font rotation and alignment
plt.xticks(rotation=45, ha='right', fontsize=12,)

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout for a more polished look
plt.tight_layout()

# Show plot
plt.show()

# plt.boxplot(df["Age"])
# # sns.stripplot(x = df["Age"], data=df, jitter=True, color='red', alpha=0.6)

# plt.title("Age Distribution")
# plt.ylabel("Age")
# plt.show()

# Convert 'Had_Heart_Attack' to categorical labels
df['Heart_Attack_Status'] = df['Had_Heart_Attack'].map({0: 'No Heart Attack', 1: 'Heart Attack'})

plt.figure(figsize=(10, 6))

# Create boxplot
boxplot = sns.boxplot(
    x='Heart_Attack_Status', 
    y='Age',
    data=df,
    palette={'No Heart Attack': 'lightgreen', 'Heart Attack': 'lightcoral'},
    width=0.5,
    linewidth=2,
    flierprops=dict(marker='o', markersize=8, markerfacecolor='grey')
)

# Add stripplot overlay
stripplot = sns.stripplot(
    x='Heart_Attack_Status',
    y='Age',
    data=df,
    jitter=True,
    alpha=0.5,
    size=6,
    palette={'No Heart Attack': 'darkgreen', 'Heart Attack': 'red'},
    edgecolor='black',
    linewidth=0.5
)

# Customize plot appearance
plt.title('Age Distribution by Heart Attack Status', fontsize=14, pad=20)
plt.xlabel('Heart Attack Occurrence', fontsize=12)
plt.ylabel('Age (Years)', fontsize=12)
plt.xticks(fontsize=11)

# # Add median value annotations
# medians = df.groupby('Heart_Attack_Status')['Age'].median()
# for xtick in boxplot.get_xticks():
#     plt.text(xtick, medians[xtick]+1, f'Median: {medians[xtick]:.0f}', 
#              ha='center', va='bottom', fontsize=11, color='black',
#              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Improve grid and layout
plt.grid(axis='y', linestyle='--', alpha=0.3)
sns.despine()
plt.tight_layout()
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Create scatter plot with enhanced features
scatter = plt.scatter(
    df["Age"], 
    df["Time_To_Heart_Attack"],
    c=df["BP_Score"],  # Color by BP_Score
    cmap="Accent",    # Color map
    alpha=0.8,         # Transparency
    s=100,             # Marker size
    edgecolor='w',     # White edges
    linewidth=0.5      # Edge width
)

# Add labels and title
plt.title("Age vs Time to Heart Attack (Colored by BP Score)", fontsize=14, pad=20)
plt.xlabel("Age (years)", fontsize=12)
plt.ylabel("Days to Heart Attack Post-Vaccination", fontsize=12)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Blood Pressure Score', fontsize=12)


# Improve tick labels
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Remove top and right spines
sns.despine()

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# sns.set(style="whitegrid")

# Plot
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Vaccine Dose', hue='Had_Heart_Attack', palette=['skyblue', 'red'])

for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(
            p.get_x() + p.get_width() / 2.,   # x position
            height + 0.5,                     # y position
            int(height),                      # value to show
            ha="center", fontsize=10
        )

# Labels
plt.title('Heart Attacks by Vaccine Dose')
plt.xlabel('Vaccine Dose (1 = First, 2 = Second, 3 = Booster)')
plt.ylabel('Number of People')
plt.legend(title='Had Heart Attack', labels=['No', 'Yes'])

plt.show()

# Filter out the rows where BP_Score is not in the range of 1 to 4
filtered_df = df[df['BP_Score'].isin([1, 2, 3, 4])]

plt.figure(figsize=(7, 5))
sns.countplot(x='BP_Score', data=filtered_df, palette='Set2')

plt.title('Count of Patients by BP Score')
plt.xlabel('BP Score (1: Normal, 2: Elevated, 3: High, 4: Very High)')
plt.ylabel('Number of Patients')

# Add bar labels
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width() / 2., p.get_height() + 1,
             int(p.get_height()), ha='center')

plt.tight_layout()
plt.show()


g = sns.lmplot(
   x='Cholesterol Level',
   y='Time_To_Heart_Attack',
   data=df[df["Had_Heart_Attack"] == 1],
   hue='Gender',
   palette='Set1',
   height=6,
   aspect=1.5,
   markers=["o", "s"]
)

# Access the underlying axes object
ax = g.ax

# Add labels and title
ax.set_title('Cholesterol Level vs Time To Heart Attack by Gender')
ax.set_xlabel('Cholesterol Level')
ax.set_ylabel('Time To Heart Attack (in days)')
ax.grid(True, linestyle='--', alpha=0.3)

# Fix legend (it's already there from hue, no need to reset manually)
# Just update the legend title if needed
# Update the legend with proper labels and title
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Female', 'Male'], title='Gender', loc='best', fontsize=10, title_fontsize=12, frameon=True, shadow=True)

plt.tight_layout()
plt.show()


px.scatter(
    df, 
    x='Age', 
    y='Time_To_Heart_Attack', 
    color='Gender', 
    color_discrete_map={1: 'yellow', 0: 'blue'},
    labels={'Gender': 'Gender (Yellow: Male, Blue: Female)'}
)

# ## ML model Work
# 


ss = StandardScaler()

cols = ['Age', 'Days_Since_Vaccination', 'BP_Score', 'Cholesterol Level', 'BMI', 'Vax_Year', 'Vax_Month']
for col in cols:
    df[col] = ss.fit_transform(df[[col]])

df.head()

numerical_features = [
    'Age',
    'Gender',                  # 0 = Female, 1 = Male
    'Vaccine Dose',             # 1=1st Dose, 2=2nd Dose, 3=Booster
    'Days_Since_Vaccination',  # Numeric
    'Vax_Year',                 # Optional but useful
    'Vax_Month',                # Optional
    'BP_Score',
    'Cholesterol Level',
    'BMI',
    'Smoking History',          # 0/1
    'Diabetes Status'           # 0/1
]



categorical_features = [
    # Pre-existing conditions
    'Pre-existing Conditions_Heart Disease',
    'Pre-existing Conditions_Hypertension',
    'Pre-existing Conditions_Multiple Conditions',
    'Pre-existing Conditions_None',
    'Pre-existing Conditions_Obesity',
    'Pre-existing Conditions_Smoking',
      
    # Location (one-hot)
    'Location_Agra',
    'Location_Bangalore',
    'Location_Bhopal',
    'Location_Chandigarh',
    'Location_Chennai',
    'Location_Delhi',
    'Location_Hyderabad',
    'Location_Indore',
    'Location_Jaipur',
    'Location_Kolkata',
    'Location_Lucknow',
    'Location_Ludhiana',
    'Location_Mumbai',
    'Location_Nagpur',
    'Location_Patna',
    'Location_Pune',
]


X = df[numerical_features + categorical_features]
y = df['Had_Heart_Attack']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

RF = RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
LG = LogisticRegression()

lg = LG.fit(X_train, y_train)
y_pred = lg.predict(X_test)

print("the accuracy score is :", accuracy_score(y_test, y_pred))
print("The Confusion Matrix is:")
print(confusion_matrix(y_test, y_pred))
print("precision score",precision_score(y_test,y_pred))
print("Recall score",recall_score(y_test,y_pred))
print("F1 is ", f1_score(y_test, y_pred)    )
print(classification_report(y_test, y_pred))

rand_model = RF.fit(X_train,y_train)
y_pred_rf = rand_model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2 = r2_score(y_test, y_pred_rf)

print("Random Forest Predictions:", y_pred_rf)
print("Random Forest RMSE:", rmse)
print("Random Forest R²:", r2)



print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-score:", f1_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

scaler = StandardScaler()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Assume 'df' is your fully preprocessed DataFrame

# --- 1. DEFINE FEATURES (UPDATED - Vax_Day removed) ---
numerical_features = [
    'Age',
    'Gender',                  # 0 = Female, 1 = Male
    'Vaccine Dose',             # 1=1st Dose, 2=2nd Dose, 3=Booster
    'Days_Since_Vaccination',  # Numeric
    'Vax_Year',                 # Optional but useful
    'Vax_Month',                # Optional
    'BP_Score',
    'Cholesterol Level',
    'BMI',
    'Smoking History',          # 0/1
    'Diabetes Status'           # 0/1
]

categorical_features = [
    # Pre-existing conditions
    'Pre-existing Conditions_Heart Disease',
    'Pre-existing Conditions_Hypertension',
    'Pre-existing Conditions_Multiple Conditions',
    'Pre-existing Conditions_None',
    'Pre-existing Conditions_Obesity',
    'Pre-existing Conditions_Smoking',
    # Location (one-hot)
    'Location_Agra',
    'Location_Bangalore',
    'Location_Bhopal',
    'Location_Chandigarh',
    'Location_Chennai',
    'Location_Delhi',
    'Location_Hyderabad',
    'Location_Indore',
    'Location_Jaipur',
    'Location_Kolkata',
    'Location_Lucknow',
    'Location_Ludhiana',
    'Location_Mumbai',
    'Location_Nagpur',
    'Location_Patna',
    'Location_Pune',
]

X = df[numerical_features + categorical_features]
y = df['Had_Heart_Attack']

# --- 2. SPLIT DATA FIRST ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. CORRECTLY FIT AND TRANSFORM THE SCALER ---
# Instantiate ONE scaler
scaler = StandardScaler()

# FIT the scaler ONLY on the training data's numerical features
scaler.fit(X_train[numerical_features])

# TRANSFORM the training data
X_train[numerical_features] = scaler.transform(X_train[numerical_features])

# TRANSFORM the test data using the SAME fitted scaler
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# --- 4. TRAIN MODELS ---
RF = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
LG = LogisticRegression(class_weight='balanced', random_state=42)

lg = LG.fit(X_train, y_train)
rand_model = RF.fit(X_train, y_train)


# --- 5. CREATE AND PREDICT ON A HIGH-RISK TEST CASE (UPDATED) ---
high_risk_row = {
    'Age': 70, 'Gender': 1, 'Vaccine Dose': 1, 'Days_Since_Vaccination': 400,
    'Vax_Year': 2022, 'Vax_Month': 6, # 'Vax_Day' removed here
    'BP_Score': 4, 'Cholesterol Level': 280, 'BMI': 32, 'Smoking History': 1, 'Diabetes Status': 1,
    'Pre-existing Conditions_Heart Disease': 1, 'Pre-existing Conditions_Hypertension': 1,
    'Pre-existing Conditions_Multiple Conditions': 0, 'Pre-existing Conditions_None': 0,
    'Pre-existing Conditions_Obesity': 0, 'Pre-existing Conditions_Smoking': 0,
    'Location_Agra': 0, 'Location_Bangalore': 0, 'Location_Bhopal': 0, 'Location_Chandigarh': 0,
    'Location_Chennai': 0, 'Location_Delhi': 1, 'Location_Hyderabad': 0, 'Location_Indore': 0,
    'Location_Jaipur': 0, 'Location_Kolkata': 0, 'Location_Lucknow': 0, 'Location_Ludhiana': 0,
    'Location_Mumbai': 0, 'Location_Nagpur': 0, 'Location_Patna': 0, 'Location_Pune': 0
}

# Convert to DataFrame
high_risk_df = pd.DataFrame([high_risk_row])

# Make sure all columns exist and are in the same order as X_train
for col in X_train.columns:
    if col not in high_risk_df.columns:
        high_risk_df[col] = 0
high_risk_df = high_risk_df[X_train.columns]

# Scale numeric features using the CORRECTLY fitted scaler
high_risk_df[numerical_features] = scaler.transform(high_risk_df[numerical_features])

# Make prediction
prediction_high = lg.predict(high_risk_df)
prediction_proba_high = lg.predict_proba(high_risk_df)

print("--- High-Risk Test Case Prediction ---")
print("Prediction (Had Heart Attack=1 / No=0):", prediction_high[0])
print("Prediction Probabilities [No Heart Attack, Heart Attack]:", prediction_proba_high[0])

low_risk_row = {
    'Age': 28, 'Gender': 0, 'Vaccine Dose': 2, 'Days_Since_Vaccination': 180,
    'Vax_Year': 2023, 'Vax_Month': 5, # 'Vax_Day' removed here
    'BP_Score': 1, 'Cholesterol Level': 165, 'BMI': 22.5, 'Smoking History': 0, 'Diabetes Status': 0,
    'Pre-existing Conditions_Heart Disease': 0, 'Pre-existing Conditions_Hypertension': 0,
    'Pre-existing Conditions_Multiple Conditions': 0, 'Pre-existing Conditions_None': 1,
    'Pre-existing Conditions_Obesity': 0, 'Pre-existing Conditions_Smoking': 0,
    'Location_Agra': 0, 'Location_Bangalore': 0, 'Location_Bhopal': 0, 'Location_Chandigarh': 0,
    'Location_Chennai': 0, 'Location_Delhi': 0, 'Location_Hyderabad': 0, 'Location_Indore': 0,
    'Location_Jaipur': 0, 'Location_Kolkata': 0, 'Location_Lucknow': 0, 'Location_Ludhiana': 0,
    'Location_Mumbai': 1, 'Location_Nagpur': 0, 'Location_Patna': 0, 'Location_Pune': 0
}

# Convert to DataFrame
low_risk_df = pd.DataFrame([low_risk_row])

# Make sure all columns exist and are in the same order as X_train
for col in X_train.columns:
    if col not in low_risk_df.columns:
        low_risk_df[col] = 0
low_risk_df = low_risk_df[X_train.columns]

# Scale numeric features using the CORRECTLY fitted scaler
low_risk_df[numerical_features] = scaler.transform(low_risk_df[numerical_features])

# Make prediction
prediction_low = lg.predict(low_risk_df)
prediction_proba_low = lg.predict_proba(low_risk_df)

print("\n--- Low-Risk Test Case Prediction ---")
print("Prediction (Had Heart Attack=1 / No=0):", prediction_low[0])
print("Prediction Probabilities [No Heart Attack, Heart Attack]:", prediction_proba_low[0])

import matplotlib.pyplot as plt
import seaborn as sns

# Set a nice style for the plots
sns.set_style("whitegrid")

# --- Plot for Random Forest Feature Importance ---
rf_importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rand_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=rf_importances.head(15), palette='viridis')
plt.title('Random Forest: Top 15 Most Important Features', fontsize=16)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()

# --- Plot for Logistic Regression Coefficients ---
# Coefficients show direction (positive or negative correlation)
lr_coefficients = pd.DataFrame({
    'feature': X_train.columns,
    'coefficient': lg.coef_[0]
}).sort_values('coefficient', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='coefficient', y='feature', data=lr_coefficients.head(15), palette='coolwarm')
plt.title('Logistic Regression: Top 15 Features (Positive Correlation)', fontsize=16)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.show()

import shap

# Create a SHAP explainer object
# We use the Random Forest model for this example
explainer = shap.TreeExplainer(rand_model)

# Calculate SHAP values for your low-risk test case
shap_values = explainer.shap_values(low_risk_df)

# The SHAP values for the positive class (Heart Attack = 1) are in shap_values[1]
shap.initjs() # This initializes the JavaScript visualization library

# Generate the force plot for the first (and only) prediction
shap.force_plot(explainer.expected_value[1], shap_values[1], low_risk_df)