import numpy as np
import matplotlib.pyplot as plt

X = np.array([[2, 50], [4, 60], [6, 65], [8, 80], [10, 85]])
y = np.array([0, 0, 0, 1, 1])

X = (X - X.mean(axis=0)) / X.std(axis=0)
X = np.c_[np.ones(X.shape[0]), X] 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(X, y, w):
    h = sigmoid(X @ w)
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

def train(X, y, lr=0.1, steps=1000):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        h = sigmoid(X @ w)
        w -= lr * X.T @ (h - y) / len(y)
    return w

weights = train(X, y)

preds = sigmoid(X @ weights) > 0.5
print("Predictions:", preds.astype(int))

z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
for i in range(len(y)):
    color = 'red' if y[i] == 0 else 'green'
    plt.scatter(X[i, 1], X[i, 2], c=color)

x_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]
plt.plot(x_vals, y_vals, label='Decision Boundary', color='blue')

plt.xlabel("Hours Studied (normalized)")
plt.ylabel("Exam Score (normalized)")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()