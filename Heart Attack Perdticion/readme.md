# Heart Attack Prediction Model

This project explores the relationship between vaccination data and heart attack occurrences. It uses machine learning models to predict the likelihood of a patient having a heart attack based on their demographic data, vaccination details, and pre-existing health conditions.

## ⚠️ Important Disclaimer: Data Bias and Model Limitations

**This model is for educational purposes only and should not be used for actual medical diagnosis.**

The dataset used in this project is highly imbalanced, meaning there are significantly more examples of patients who did not have a heart attack than those who did. This imbalance can introduce significant bias into the model.

### How this bias affects the model:
- The model may learn to prioritize accuracy by simply predicting "No Heart Attack" for most cases.
- It might perform poorly on identifying patients who are genuinely at risk (low recall for the positive class).
- The model may latch onto non-intuitive or spurious correlations in the minority class, leading to incorrect or unexpected predictions (e.g., predicting a heart attack for a low-risk patient).

**Therefore, while this project serves as a good exercise in machine learning techniques, its predictions are not reliable enough for real-world application.**

## Dataset

The project uses a dataset named `heart_attack_vaccine_data.csv`. The data includes the following key information:

- **Patient Demographics**: Age, Gender, Location  
- **Vaccination Details**: Vaccination Date, Vaccine Dose (1st, 2nd, Booster)  
- **Health Metrics**: Blood Pressure, Cholesterol Level, BMI, Smoking History, Diabetes Status  
- **Pre-existing Conditions**: Heart Disease, Hypertension, Obesity, etc.  
- **Outcome**: Heart Attack Date, Severity, Outcome (Survived/Not Survived)

## Methodology

### 1. Data Preprocessing & Feature Engineering
- **Data Cleaning**: Duplicates were removed, and missing values in columns like `Severity` and `Outcome` were filled.
- **Categorical Encoding**: Categorical features were converted to numerical format using mapping (e.g., Blood Pressure: `'Normal' → 1`, `'High' → 3`) and One-Hot Encoding (e.g., Location, Pre-existing Conditions).
- **Date/Time Features**: Vaccination Date was converted to datetime, and new features like `Vax_Year`, `Vax_Month`, and `Days_Since_Vaccination` were created.
- **Target Variable**: The target variable `Had_Heart_Attack` was created as a binary flag (`1` if Heart Attack Date is not null, `0` otherwise).
- **Feature Scaling**: Numerical features were scaled using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1, which is crucial for models like Logistic Regression.

### 2. Model Training
- **Features**: The model was trained on a combination of numerical and categorical features. Critically, data leakage was avoided by removing features like `Outcome` that would only be known after a heart attack.
- **Class Imbalance Handling**: To address the data imbalance, the `class_weight='balanced'` parameter was used in the models. This tells the model to give more importance to the minority class (heart attack cases).
- **Models Used**:
  - **Logistic Regression**: A linear model used as a baseline.
  - **Random Forest Classifier**: An ensemble model that generally performs well on tabular data.

### 3. Model Evaluation
The models were evaluated on a hold-out test set using the following metrics:
- **Accuracy**: The overall percentage of correct predictions.
- **Precision**: Of all the patients the model predicted would have a heart attack, how many actually did?
- **Recall**: Of all the patients who actually had a heart attack, how many did the model correctly identify?
- **F1-Score**: The harmonic mean of Precision and Recall, providing a single metric for model performance.

## How to Use the Model: Test Case Examples

Below are two examples of how to use the trained model to make predictions on new patient data.

### Example 1: High-Risk Patient

This patient has multiple risk factors, including advanced age, high cholesterol, and diabetes. The model correctly predicts a high risk of a heart attack.

```python
# Test case for a high-risk patient
high_risk_row = {
    'Age': 70, 'Gender': 1, 'Vaccine Dose': 1, 'Days_Since_Vaccination': 400,
    'Vax_Year': 2022, 'Vax_Month': 6, 'BP_Score': 4, 'Cholesterol Level': 280,
    'BMI': 32, 'Smoking History': 1, 'Diabetes Status': 1,
    'Pre-existing Conditions_Heart Disease': 1, 'Pre-existing Conditions_Hypertension': 1,
    'Pre-existing Conditions_Multiple Conditions': 0, 'Pre-existing Conditions_None': 0,
    'Pre-existing Conditions_Obesity': 0, 'Pre-existing Conditions_Smoking': 0,
    'Location_Agra': 0, 'Location_Bangalore': 0, 'Location_Bhopal': 0,
    'Location_Chandigarh': 0, 'Location_Chennai': 0, 'Location_Delhi': 1,
    'Location_Hyderabad': 0, 'Location_Indore': 0, 'Location_Jaipur': 0,
    'Location_Kolkata': 0, 'Location_Lucknow': 0, 'Location_Ludhiana': 0,
    'Location_Mumbai': 0, 'Location_Nagpur': 0, 'Location_Patna': 0, 'Location_Pune': 0
}

# Expected Output:
# Prediction (Had Heart Attack=1 / No=0): 1
# Prediction Probabilities [No Heart Attack, Heart Attack]: [0. 1.]
