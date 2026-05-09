# 🏎️ F1 Canadian Grand Prix Podium Predictor

A data science project that predicts the top podium contenders for the **2026 Formula 1 Canadian Grand Prix** using current season standings, historical circuit performance, and driver reliability data — built entirely with Pandas, NumPy, and Scikit-learn.

---

## 📌 Problem Statement

Formula 1 race outcomes depend on a combination of current form, circuit-specific history, car reliability, and recent consistency. This project builds a transparent, explainable weighted scoring model that ranks drivers by their podium probability at Circuit Gilles-Villeneuve, Montreal — one of the most unpredictable tracks on the F1 calendar.

---

## 📂 Dataset

- **Source:** Manually constructed from real 2026 F1 season data
- **Size:** 10 drivers (top points scorers in the 2026 championship)
- **Features:** Current season points, historical Canadian GP wins, historical Canadian GP podiums, average finishing position in 2026, DNF count in 2026

> **Note:** Data reflects the 2026 F1 season standings after 4 races, and each driver's complete career record at the Canadian GP.

---

## 🛠️ Tech Stack

| Library | Usage |
|---|---|
| `Pandas` | Dataset construction, feature engineering, groupby analysis, output formatting |
| `NumPy` | Vectorized feature creation, `np.where` for conditional scoring, matrix multiplication |
| `Scikit-learn` | `MinMaxScaler` for feature normalization |

---

## ⚙️ Project Workflow

### 1. Dataset Construction
- Manually built a structured Pandas DataFrame with real 2026 season standings
- Features captured both **current form** (season points, avg finish) and **circuit-specific history** (Canadian GP wins and podiums)

### 2. Feature Engineering

**Canadian Podium Rate**
```python
df['Canadian_Podium_Rate'] = df['Canadian_GP_Podiums'] / 10
```
Converts raw podium counts into a normalized rate — Hamilton's 9 career podiums at Canada become a 0.9 rate, reflecting his exceptional affinity for this circuit.

**Reliability Score using `np.where`**
```python
df['Reliability_Score'] = np.where(df['DNF_Count_2026'] == 0, 1.0,
                           np.where(df['DNF_Count_2026'] == 1, 0.7, 0.4))
```
A vectorized if-elif-else applied across the entire column simultaneously — penalizing drivers with mechanical failures since a DNF earns zero championship points regardless of pace.

**Inverse Average Finish**
```python
df['Inverse_Avg_Finish'] = 1 / df['Avg_Finish_2026']
```
In F1, a lower finishing position is better (P1 > P10). Inverting the value ensures higher numbers consistently represent better performance across all features.

### 3. Feature Normalization
```python
scaler = MinMaxScaler()
df_scaled[features] = scaler.fit_transform(df[features])
```
All four features are scaled to a 0–1 range using MinMaxScaler so no single feature dominates the final score purely due to its numerical magnitude.

### 4. Weighted Scoring Model
```python
weights = np.array([0.40, 0.25, 0.20, 0.15])
df['Podium_Score'] = df_scaled[features].values @ weights
```
A NumPy dot product combines all normalized features into a single composite score using domain-driven weights.

| Feature | Weight | Reasoning |
|---|---|---|
| `Season_Points` | 40% | Current form is the strongest predictor of race pace |
| `Canadian_Podium_Rate` | 25% | Circuit-specific history matters significantly in F1 |
| `Reliability_Score` | 20% | One DNF eliminates a driver entirely |
| `Inverse_Avg_Finish` | 15% | Consistent finishing reflects overall car competitiveness |

### 5. Ranking
```python
df_sorted = df.sort_values('Podium_Score', ascending=False).reset_index(drop=True)
df_sorted['Predicted_Rank'] = df_sorted.index + 1
```

---

## 📊 Results

| Predicted Rank | Driver | Team | Season Points | Podium Score |
|---|---|---|---|---|
| 1 | Antonelli | Mercedes | 72 | 0.847 |
| 2 | Russell | Mercedes | 63 | 0.761 |
| 3 | Norris | McLaren | 54 | 0.643 |
| 4 | Hamilton | Ferrari | 28 | 0.591 |
| 5 | Piastri | McLaren | 48 | 0.534 |

> **Interesting insight:** Hamilton ranks 4th despite sitting 5th in the championship — his 9 career podiums at Circuit Gilles-Villeneuve give him a significant circuit-specific boost that the points standings alone don't capture.

---

## 🔍 Key Design Decisions

**Why a weighted scoring model instead of a ML classifier?**
The dataset contains only 10 drivers — far too small to train a meaningful supervised model. A weighted scoring system is more appropriate, fully transparent, and directly explainable. With years of historical race data across all drivers, a proper ML model (e.g. Random Forest or Gradient Boosting) would be the natural next step.

**Why MinMaxScaler over StandardScaler?**
StandardScaler normalizes based on mean and standard deviation, assuming roughly normal distribution. MinMaxScaler compresses everything into a fixed 0–1 range, which is ideal when combining features into a single composite score — ensuring no feature dominates due to scale differences alone.

**Why `np.where` for the reliability score?**
`np.where` is vectorized — it applies the conditional logic across the entire column in one C-level operation rather than iterating row by row in Python. It's significantly faster and more concise than an equivalent loop or `apply()`.

**Why invert average finish position?**
Raw finishing positions are inversely related to performance — P1 is better than P10. Inverting ensures all features point in the same direction: higher values always mean better performance, making the weighted combination logically consistent.

---

## 📁 Project Structure

```
f1-canadian-gp-predictor/
│
├── notebook/
│   └── f1_canadian_gp_predictor.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/f1-canadian-gp-predictor.git
cd f1-canadian-gp-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook notebook/f1_canadian_gp_predictor.ipynb
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

- Weights are domain-driven and manually assigned — a larger historical dataset would allow learning optimal weights through regression, making the model data-driven rather than assumption-driven.
- The model does not account for qualifying results, weather conditions, safety car probability, or pit stop strategy — all of which heavily influence F1 race outcomes.
- Hamilton's high circuit-specific score reflects his career record but not his current car's competitiveness at Ferrari in 2026 — a limitation of separating historical and current-season data.
- Future improvements: integrating qualifying lap time data, expanding to all 20 drivers, and building a Streamlit dashboard for live race weekend predictions.

---
