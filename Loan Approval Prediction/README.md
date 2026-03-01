# Loan Approval Prediction

## 📌 Project Overview
This project is a **machine learning-based solution** to predict whether a loan application will be approved or not. The goal is to help financial institutions make data-driven decisions, reduce risks, and automate loan processing.

- **Problem Type:** Classification  
- **Target Variable:** `Loan_Status` (Approved / Not Approved)  
- **Dataset:** Kaggle dataset (link if public)  

---

## 🧠 Objective
The main objectives of this project are:

1. Analyze and clean loan application data.  
2. Handle missing values, outliers, and class imbalance.  
3. Build and compare multiple machine learning models.  
4. Identify the best model for predicting loan approval.  

---

## 📊 Dataset Details
The dataset contains information about loan applicants, including:

| Feature | Description |
|---------|-------------|
| `Gender` | Male / Female |
| `Married` | Yes / No |
| `Dependents` | Number of dependents |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Yes / No |
| `ApplicantIncome` | Applicant's income |
| `CoapplicantIncome` | Co-applicant's income |
| `LoanAmount` | Loan amount requested |
| `Loan_Amount_Term` | Term of the loan in months |
| `Credit_History` | 1 = Meets guidelines, 0 = Does not meet |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | Target variable |

---

## 🔧 Data Preprocessing
- Handling missing values in numeric and categorical columns.  
- Converting invalid or negative values to proper format.  
- Encoding categorical variables (Label Encoding / One-Hot Encoding).  
- Feature scaling (StandardScaler) for distance-based models like SVM.  
- Handling class imbalance with techniques like `class_weight='balanced'`.  

---

## 🛠️ Models Implemented
The following models were implemented and evaluated:

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|---------|----------|---------|
| Logistic Regression | 0.846 | 0.728 | 0.863 |
| Decision Tree | 0.932 | 0.851 | 0.901 |
| Random Forest | 0.964 | 0.919 | 0.937 |
| Gradient Boost | 0.963 | 0.917 | 0.936 |
| Support Vector Machine | 0.882 | 0.775 | 0.882 |

**Best Model:** Random Forest  

> Tree-based ensemble methods outperformed linear and SVM models due to capturing non-linear relationships and feature interactions.

---

## 📈 Key Findings
- Logistic Regression and SVM performed poorly initially due to lack of scaling and hyperparameter tuning.  
- After scaling and tuning, both models improved significantly.  
- Random Forest achieved the **highest accuracy, F1 Score, and ROC-AUC**, making it the best model for this task.  
- Feature interactions and non-linear patterns in the dataset make tree-based models more suitable.

---

## 💻 How to Run
1. Clone the repository:
```bash
git clone https://github.com/your-username/loan-approval-prediction.git
