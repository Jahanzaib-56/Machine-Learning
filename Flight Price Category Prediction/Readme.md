# ✈️ Flight Price Category Predictor

A machine learning project that predicts whether a flight price is **Cheap**, **Moderate**, or **Expensive** based on flight details such as airline, route, departure time, number of stops, and days left before departure.

---

## 📌 Problem Statement

Flight prices fluctuate heavily based on multiple factors. This project builds a classification pipeline that categorizes flight prices into three tiers using real-world data, helping travelers make informed booking decisions.

---

## 📂 Dataset

- **Source:** [EaseMyTrip Flight Dataset — Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
- **Size:** ~300,000 rows
- **Features:** Airline, source city, destination city, departure time, arrival time, stops, duration, days left before departure, class, price

---

## 🛠️ Tech Stack

| Library | Usage |
|---|---|
| `Pandas` | Data loading, EDA, feature engineering, encoding |
| `NumPy` | Vectorized operations, percentile-based label creation |
| `Scikit-learn` | Pipeline, preprocessing, model training, evaluation |

---

## ⚙️ Project Workflow

### 1. Exploratory Data Analysis
- Checked dataset shape, data types, and null values
- Analyzed average price per airline using `groupby`
- Identified key patterns between `days_left`, `stops`, and price

### 2. Feature Engineering
- Converted `duration` from string format (`"2h 30m"`) to total minutes (integer)
- Created price category labels dynamically using `np.percentile`:
  - **Cheap** → bottom 33rd percentile
  - **Moderate** → 33rd to 66th percentile
  - **Expensive** → top 33rd percentile
- Encoded target variable using `OrdinalEncoder` with explicit ordering: `Cheap=0, Moderate=1, Expensive=2`
- One-hot encoded categorical features using `pd.get_dummies()`
- Converted entire DataFrame to numeric using `pd.to_numeric(errors='coerce')`

### 3. Modeling Pipeline
```
SimpleImputer → StandardScaler → RandomForestClassifier
```
Built using `sklearn.pipeline.Pipeline` to prevent data leakage and ensure clean, reproducible preprocessing.

### 4. Evaluation
- `classification_report` — Precision, Recall, F1 per class
- `confusion_matrix` — Error analysis across categories
- `cross_val_score` (5-fold) — Robust generalization estimate
- Feature importance — Identified top predictive features

---

## 📊 Results

| Metric | Score |
|---|---|
| Cross-Validation Accuracy (5-fold) | ~0.94 |
| Weighted F1-Score | ~0.94 |

**Top predictive features:**
1. `days_left` — strongest signal; prices spike close to departure
2. `stops` — non-stop flights command premium pricing
3. `duration` — longer flights tend to cost more
4. `class` — business class significantly inflates price
5. `airline` — certain carriers consistently price higher

---

## 🔍 Key Design Decisions

**Why OrdinalEncoder over LabelEncoder for the target?**
LabelEncoder assigns alphabetical integer order (Cheap=0, Expensive=1, Moderate=2), which implies Expensive is between Cheap and Moderate — semantically incorrect. OrdinalEncoder allows explicit ordering that reflects the true price hierarchy.

**Why RandomForestClassifier?**
Random Forest makes decisions through threshold-based splits (Gini impurity / information gain), not distance or gradient calculations. This makes it naturally robust on one-hot encoded data and large mixed-feature datasets without requiring heavy hyperparameter tuning for a strong baseline.

**Why percentile-based price categories?**
Hardcoding price thresholds would be dataset-specific and arbitrary. Using `np.percentile` creates balanced classes dynamically, making the classification problem well-defined and the evaluation metrics meaningful.

**Why Pipeline?**
The `SimpleImputer` and `StandardScaler` inside the Pipeline are fitted only on training data. This prevents data leakage — test set statistics never influence the preprocessing steps.

---

## 📁 Project Structure

```
flight-price-predictor/
│
├── data/
│   └── flight_data.csv
│
├── notebook/
│   └── flight_price_predictor.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/flight-price-predictor.git
cd flight-price-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebook/flight_price_predictor.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
jupyter
```

---

## 🧠 Learnings & Limitations

- A 0.94 cross-validation score is strong but partly reflects the high predictive power of `days_left`, which is a near-direct proxy for price. Real-world generalization should be validated on future unseen flight data.
- Price categories are relative to this dataset's distribution — the model would need retraining on different markets or time periods.
- Future improvements: hyperparameter tuning with `GridSearchCV`, Streamlit deployment for live predictions, and expanding to regression for exact price prediction.

---
