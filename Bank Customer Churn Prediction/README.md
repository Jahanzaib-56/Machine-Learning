# Bank Customer Churn Prediction

## Overview
This project aims to predict whether a bank customer is likely to churn (leave the bank) based on their demographic and account-related information. By identifying at-risk customers, banks can take proactive retention measures, improving customer satisfaction and reducing revenue loss.

The project demonstrates the implementation of a full machine learning pipeline, including data preprocessing, model training, evaluation, and interpretation.

---

## Dataset
The dataset contains customer demographic information, account details, and whether they churned. Key features include:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

The target variable is:
- `Exited` (1 if the customer churned, 0 otherwise)

---

## Preprocessing
- Categorical variables were encoded (e.g., Geography, Gender)
- Features were scaled using **StandardScaler**
- Data was split into training and testing sets **before scaling** to avoid data leakage

---

## Models Used
Two supervised classification models were implemented and evaluated:

1. **Logistic Regression**  
   - Achieved **99.9% test accuracy**  
   - Minimal difference between training and testing scores indicates excellent generalization  
   - Strong performance suggests dataset is highly linearly separable

2. **K-Nearest Neighbors (KNN)**  
   - Achieved **99.2% test accuracy**  
   - Perfect precision but slightly lower recall, indicating sensitivity to local data distribution

---

## Evaluation Metrics
- Accuracy
- F1 Score
- Precision
- Recall
- Confusion Matrix

The near-perfect metrics for both models demonstrate highly reliable predictive performance.

---

## Insights
- Logistic Regression outperforming KNN indicates that the dataset's decision boundaries are approximately linear.
- Both models generalize well, with minimal overfitting.
- KNN’s slightly lower recall highlights areas where local patterns could lead to missed predictions.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bank-customer-churn-prediction.git
