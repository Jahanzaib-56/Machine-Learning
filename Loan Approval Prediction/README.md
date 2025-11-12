🧠 Project Overview

This project aims to predict whether a loan application should be approved or not using Machine Learning models.
The idea is to help financial institutions make faster, data-driven decisions when assessing loan requests.

I collected a Loan Approval Dataset from Kaggle, cleaned and preprocessed it, and then trained several classification models to find the most accurate one.

📊 Dataset

Source: Kaggle - Loan Prediction Dataset

Description:
The dataset contains information about loan applicants, such as:

Applicant income

Loan amount

Credit history

Employment years

Loan approval status (Target variable)

🧹 Data Preparation

Before training, the dataset was carefully prepared:

✅ Handled missing values

✅ Scaled numerical features for consistency

✅ Split data into training and testing sets

🤖 Machine Learning Models

I implemented and compared the following models:

Model	Description
Logistic Regression (LR)	Simple linear classifier for binary outcomes
Decision Tree Classifier (DTC)	Tree-based model capturing non-linear relations
Support Vector Classifier (SVC)	Finds optimal decision boundaries in high-dimensional space
K-Nearest Neighbors (KNC)	Classifies based on the majority vote of nearest neighbors

Each model was trained, tested, and evaluated using metrics like Accuracy and Confusion Matrix.

🏆 Model Performance
Model	Training Accuracy	Test Accuracy
Logistic Regression	97.4%	95.7%
Decision Tree	100%	100% (Over-fitting)
KNN	99.0%	98.0%
SVM	99.1%	97.5%

📈 Best Performing Model: Logistic Regression

⚙️ Technologies Used

- Python 🐍

- NumPy

- Pandas

- Scikit-learn

- Matplotlib / Seaborn (for visualization)
