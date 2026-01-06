### 📊 Loan Approval & Risk Assessment ML Project  
Building predictive models for financial risk scoring and automated loan decisions  

**Technologies**: Python, Pandas, Scikit-learn, XGBoost  
**Author**: Hassaan Shahid  

---

### 📌 Project Overview  
This project focuses on building predictive machine learning models for loan approval and risk scoring. The analysis leverages a synthetic dataset of **20,000 records** with **36 features**, including demographic information, financial data, credit history, and loan details.

---

### 🎯 Objectives  
- **Binary Classification**: Predict whether a loan application will be approved (`0`: Not Approved, `1`: Approved).  
- **Regression**: Estimate a continuous **Risk Score** for each applicant.  
- **Feature Analysis**: Identify key drivers behind approval decisions and risk evaluation.

---

### 🛠️ Dataset Description  
The dataset encompasses a wide range of financial and personal attributes:

| Category              | Features |
|----------------------|----------|
| **Demographics**     | Age, Employment Status, Education Level, Marital Status |
| **Financial Metrics**| Annual Income, Monthly Debt Payments, Savings/Checking Balances, Net Worth |
| **Credit Information**| Credit Score, Open Credit Lines, Debt-to-Income Ratio, Previous Defaults, Payment History |
| **Loan Details**     | Loan Amount, Duration, Purpose, Base & Applied Interest Rates |
| **Temporal Features**| Application Date, Extracted Month & Year |

---

### 🎯 Target Variables  
- `LoanApproved`: Binary classification target (`0` or `1`)  
- `RiskScore`: Continuous regression target

---

### 🔄 Data Preprocessing  
- **Missing Values**: Imputed using `SimpleImputer`.  
- **Numerical Scaling**: Standardized using `StandardScaler`.  
- **Categorical Encoding**:  
  - *Ordinal*: Employment Status, Education Level  
  - *One-Hot*: Loan Purpose, Home Ownership  
- **Feature Engineering**: Extracted `Year` and `Month` from `ApplicationDate`.  
- **Class Imbalance**: Handled via `class_weight='balanced'` and `scale_pos_weight`.

---

### 🤖 Machine Learning Models  
1. **Random Forest Classifier** – Robust, handles non-linearity, provides feature importance.  
2. **Logistic Regression** – Baseline model (with scaled features).  
3. **XGBoost Classifier** – High performance, tuned for F1-score, optimized for imbalance.

---

### 📊 Evaluation Metrics  
- Precision, Recall, F1-score  
- Accuracy  
- ROC-AUC  
- Confusion Matrix  

#### 📉 Confusion Matrix (XGBoost)  
| Actual \ Predicted | 0 (Rejected) | 1 (Approved) |
|--------------------|--------------|--------------|
| **0 (Rejected)**   | 2967         | 77           |
| **1 (Approved)**   | 195          | 761          |

---

### 💡 Key Insights  
- Top predictors: **Interest Rate**, **Credit Score**, **Annual Income**.  
- Ordinal features (e.g., Education, Employment) significantly influence decisions.  
- Proper imbalance handling improves recall for approved loans (reduces false negatives).

---
