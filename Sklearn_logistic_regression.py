import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = np.array([[2, 50], [4, 60], [6, 65], [8, 80], [10, 85]])
y = np.array([0, 0, 0, 1, 1])

X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

model = LogisticRegression()
model.fit(X_scaled, y)

sk_preds = model.predict(X_scaled)
print("Sklearn Predictions:", sk_preds)

print("Accuracy:", accuracy_score(y, sk_preds))

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

print("\nConfusion Matrix:\n", confusion_matrix(y, sk_preds))

print("\nClassification Report:\n", classification_report(y, sk_preds))

plt.figure(figsize=(6, 4))
for i in range(len(y)):
    color = 'red' if y[i] == 0 else 'green'
    plt.scatter(X_scaled[i, 0], X_scaled[i, 1], c=color)

x_vals = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
y_vals = -(model.intercept_[0] + model.coef_[0][0] * x_vals) / model.coef_[0][1]
plt.plot(x_vals, y_vals, label='Decision Boundary', color='blue')

plt.xlabel("Hours Studied (normalized)")
plt.ylabel("Exam Score (normalized)")
plt.title("Sklearn Logistic Regression Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()