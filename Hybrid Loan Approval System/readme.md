
# 🏦 Loan Approval System: Hybrid Unsupervised + Supervised ML

> **Advanced Risk-Tiered Loan Approval Using Cluster-Specific Classifiers**  
> A production-grade machine learning system that segments applicants by risk and applies specialized models for higher precision and business alignment.

---

## 📌 Overview

Traditional loan approval models treat all applicants the same — but banks don’t. High-income customers with excellent credit are evaluated differently than low-income, high-debt applicants.

This project implements a **hybrid ML architecture** that:
1. **Discovers natural customer segments** using K-Means clustering
2. **Maps clusters to business risk tiers** (Low / Medium / High)
3. **Trains specialized models per tier**:
   - **Low Risk**: Logistic Regression (interpretable, stable)
   - **Medium Risk**: Random Forest (robust, handles complexity)
   - **High Risk**: XGBoost (maximizes performance on hard cases)
4. **Routes new applicants** to the correct model at inference time

✅ **Result**: Higher precision in approvals, better risk control, and explainable decisions — all while reusing robust preprocessing pipelines.

---

## 🧠 Why This Approach?

| Traditional Model | Hybrid Model |
|------------------|--------------|
| One model for all customers | Specialized models per risk segment |
| Same logic for $200k and $20k incomes | Tiered decision logic (like real banks) |
| Global class balancing | Per-tier imbalance handling |
| Black-box predictions | Explainable per segment |

> 💡 **Real-World Analogy**:  
> Just as hospitals use pediatricians for children and cardiologists for elderly patients, this system uses the right "model specialist" for each customer type.

---

## 🔁 End-to-End Workflow

# Risk-Tiered Credit Scoring Workflow

1. **Raw Data**<br>
   ↓
2. **EDA + Feature Engineering**
   - Handle missing values  
   - Encode categorical variables  
   - Create time-based features  
   - Analyze feature distributions  
   ↓
3. **Unsupervised Learning (K-Means Clustering)**
   - Preprocess with scaling (e.g., StandardScaler)  
   - Cluster applicants into 3 segments  
   ↓
4. **Risk Tier Mapping (Business Interpretation)**
   - Cluster 2 → Tier 0 (Low Risk)  
   - Cluster 0 → Tier 1 (Medium Risk)  
   - Cluster 1 → Tier 2 (High Risk)  
   ↓
5. **Tier-Specific Supervised Models**
   - **Tier 0 (Low):** Logistic Regression  
   - **Tier 1 (Medium):** Random Forest  
   - **Tier 2 (High):** XGBoost  
   - Each model trained **only** on its assigned segment  
   - Class imbalance addressed per tier (e.g., SMOTE, class weights)  
   ↓
6. **Hybrid Inference Engine**
   1. Preprocess new applicant data  
   2. Assign to risk tier using K-Means clusterer  
   3. Route to the corresponding supervised model  
   4. Return final prediction (e.g., approve/deny, probability)  
   ↓
7. **Evaluation vs Global Models**
   - Metrics: Accuracy, Precision, Recall, F1-score  
   - Per-tier performance breakdown  
   - Business impact analysis (e.g., approval rates, default reduction)



---

## 📊 Key Results

### Cluster Profile (After Mapping)

| Risk Tier             | Approval Rate | Count | Avg Credit Score | Avg Income ($) | Avg DTI |
|:---------------------:|:-------------:|:-----:|:----------------:|:--------------:|:-------:|
| **0** (Low Risk)      | 76.1%         | 2,704 | 585.1            | 126,398        | 28.4%   |
| **1** (Medium Risk)   | 20.4%         | 7,397 | 596.4            | 47,043         | 28.9%   |
| **2** (High Risk)     | 4.3%          | 5,899 | 534.3            | 43,981         | 28.1%   |

### Model Comparison (Test Set)

| Model                  | Accuracy | Precision | Recall | F1     |
|:----------------------:|:--------:|:---------:|:------:|:------:|
| **Hybrid System**      | 0.9140   | **0.9334**| 0.6893 | 0.7930 |
| Logistic (Global)      | 0.9597   | 0.8746    | 0.9707 | 0.9202 |
| Random Forest (Global) | 0.9320   | 0.9081    | 0.7960 | 0.8484 |
| XGBoost (Global)       | 0.9555   | 0.8791    | 0.9435 | 0.9102 |

> 🔍 **Insight**:  
> The hybrid system achieves **highest precision (93.3%)** — meaning when it approves a loan, it’s **very likely to be repaid**. This reduces bad debt, even at the cost of lower recall (fewer total approvals). In banking, **precision often matters more than recall** for high-risk segments.

---

## 🛠️ Technical Highlights

### Preprocessing Pipeline
- **Modular design** with `ColumnTransformer`
- Separate pipelines for:
  - **Distance-based models** (with `StandardScaler`)
  - **Tree-based models** (median imputation, no scaling)
- Handles missing values, encoding, and scaling in one fit

### Class Imbalance Handling
- **Per-cluster balancing**: Critical because high-risk cluster has only **4.3% approvals**
- `scale_pos_weight` computed **individually for each cluster**

### Reproducibility
- Fixed `random_state=42` everywhere
- Stratified train/test split
- Full pipeline serialization ready

---

## 📁 Project Structure
