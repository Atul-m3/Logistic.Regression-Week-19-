import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

X = np.array([[2, 50], [4, 60], [6, 65], [8, 80], [10, 85]])
y = np.array([0, 0, 0, 1, 1])

X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
X_manual = np.c_[np.ones(X_scaled.shape[0]), X_scaled]  # For manual model

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, lr=0.1, steps=1000):
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        h = sigmoid(X @ w)
        w -= lr * X.T @ (h - y) / len(y)
    return w

weights = train(X_manual, y)
manual_preds = sigmoid(X_manual @ weights) > 0.5

model = LogisticRegression()
model.fit(X_scaled, y)
sk_preds = model.predict(X_scaled)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
for i in range(len(y)):
    color = 'red' if y[i] == 0 else 'green'
    plt.scatter(X_scaled[i, 0], X_scaled[i, 1], c=color)
x_vals = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
y_vals = -(weights[0] + weights[1] * x_vals) / weights[2]
plt.plot(x_vals, y_vals, label='Manual Boundary', color='blue')
plt.title("Manual Logistic Regression")
plt.xlabel("Hours Studied (norm)")
plt.ylabel("Exam Score (norm)")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
for i in range(len(y)):
    color = 'red' if y[i] == 0 else 'green'
    plt.scatter(X_scaled[i, 0], X_scaled[i, 1], c=color)
y_vals_sk = -(model.intercept_[0] + model.coef_[0][0] * x_vals) / model.coef_[0][1]
plt.plot(x_vals, y_vals_sk, label='Sklearn Boundary', color='purple')
plt.title("Sklearn Logistic Regression")
plt.xlabel("Hours Studied (norm)")
plt.ylabel("Exam Score (norm)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

ConfusionMatrixDisplay.from_predictions(y, manual_preds, display_labels=["Fail", "Pass"], cmap="Blues")
plt.title("Manual Model Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_predictions(y, sk_preds, display_labels=["Fail", "Pass"], cmap="Greens")
plt.title("Sklearn Model Confusion Matrix")
plt.show()

print("Manual Model Report:\n", classification_report(y, manual_preds))
print("Sklearn Model Report:\n", classification_report(y, sk_preds))