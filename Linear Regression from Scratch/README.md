# Linear Regression from Scratch (Python)

## 📌 Overview

This project implements **Simple Linear Regression** from scratch using **NumPy**, without relying on machine learning libraries like scikit-learn. The model is trained using **Gradient Descent** to minimize the **Mean Squared Error (MSE)**.

It demonstrates the core mathematical intuition behind linear regression and optimization.

---

## ⚙️ Features

* Manual implementation of:

  * Mean Squared Error (MSE)
  * Gradient Descent optimization
* Adjustable hyperparameters:

  * Learning rate (`alpha`)
  * Number of iterations
* Works on CSV data input
* Fully transparent training process

---

## 📂 Project Structure

```
.
├── data.csv              # Dataset file (X, Y values)
├── linear_regression.py  # Main implementation
└── README.md            # Project documentation
```

---

## 🧠 How It Works

### 1. Model Equation

The hypothesis function:

```
y = m * x + b
```

Where:

* `m` = slope
* `b` = intercept

---

### 2. Loss Function (MSE)

The error is calculated using:

```
MSE = (1/N) * Σ(y - (mx + b))²
```

---

### 3. Gradient Descent

Parameters are updated iteratively:

```
b = b - α * ∂J/∂b
m = m - α * ∂J/∂m
```

Where:

* `α` is the learning rate
* Gradients are computed for each iteration

---

## 🚀 How to Run

### 1. Install Dependencies

Make sure you have NumPy installed:

```bash
pip install numpy
```

### 2. Prepare Dataset

Create a `data.csv` file with two columns:

```
x1,y1
x2,y2
x3,y3
...
```

Example:

```
1,2
2,4
3,6
4,8
```

---

### 3. Run the Script

```bash
python linear_regression.py
```

---

## 📊 Output

The program will print:

* Initial values of `m`, `b`, and error
* Final optimized values after training
* Final error

Example:

```
Starting grad_descent at b = 0, m = 0, error = ...

Running.........

After 1000 iterations, b = ..., m = ..., error = ...
```

---

## 🔧 Hyperparameters

| Parameter       | Description                   | Default Value |
| --------------- | ----------------------------- | ------------- |
| `alpha`         | Learning rate                 | 0.0001        |
| `num_iteration` | Number of training iterations | 1000          |
| `initial_m`     | Initial slope                 | 0             |
| `initial_b`     | Initial intercept             | 0             |

---

## 📈 Possible Improvements

* Vectorize operations (remove loops for efficiency)
* Add feature scaling for faster convergence
* Plot regression line using matplotlib
* Extend to multiple linear regression
* Add convergence stopping criteria

---

## 🎯 Learning Outcomes

* Understanding how linear regression works internally
* Learning gradient descent optimization
* Building ML algorithms from first principles

---

## 📜 License

This project is open-source and free to use for educational purposes.

---
