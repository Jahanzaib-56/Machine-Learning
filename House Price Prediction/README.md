# 🏠 House Price Prediction using Linear Regression

## 📌 Project Overview

This project focuses on predicting house prices using a **Linear Regression model**. The goal is to understand how different features—especially **square footage**—influence house prices and to build a reliable predictive model.

---

## 🎯 Objectives

* Perform data preprocessing and cleaning
* Explore relationships between features and target variable
* Train a Linear Regression model
* Evaluate model performance using multiple metrics
* Validate results using cross-validation

---

## 📊 Dataset Insights

* A very strong linear relationship was observed between **Square Footage** and **House Price**
* Correlation score between these features: **0.99**
* This indicates that house prices in the dataset are highly dependent on square footage

---

## 🔍 Exploratory Data Analysis

* Heatmap revealed strong feature correlation
* Scatter plot of:

  * `square_footage` vs `house_price`
  * Showed an almost perfect straight line
* This confirmed that the dataset follows a near-linear pattern

---

## 🤖 Model Used

* **Linear Regression**

---

## 📈 Model Performance

### Evaluation Metrics:

* **MAE (Mean Absolute Error):** 8174.58
* **MSE (Mean Squared Error):** 101,434,798.50
* **RMSE (Root Mean Squared Error):** 10071.48
* **R² Score:** 0.9984

---

## 🔁 Cross-Validation Results (5-Fold)

### R² Scores:

```
[0.99846, 0.99849, 0.99850, 0.99846, 0.99854]
```

**Mean R²:** 0.99849

### RMSE Scores:

```
[9794.55, 10021.60, 10113.08, 9815.73, 9429.98]
```

**Mean RMSE:** 9834.99

---

## ✅ Key Insights

* The model achieves **extremely high accuracy** due to a strong linear relationship in the dataset
* Cross-validation confirms that the model is:

  * Stable
  * Generalizable
  * Not overfitting
* **Square Footage** is the dominant feature influencing house prices

---

## ⚠️ Important Note

The exceptional performance is mainly due to the simplicity of the dataset:

* Very high correlation between input and target
* Low noise and variability

In real-world scenarios, house pricing is influenced by multiple complex factors such as:

* Location
* Neighborhood
* Condition
* Market trends

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib / Seaborn
* Scikit-learn

---

## 🚀 How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/house-price-prediction.git
```

2. Navigate to the project folder:

```
cd house-price-prediction
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the notebook or script

---

## 📌 Conclusion

This project demonstrates how powerful simple models can be when the underlying data has strong patterns. Linear Regression performs exceptionally well when the relationship between features and target is highly linear.

---

## 📬 Contact

Feel free to reach out for feedback or collaboration!
