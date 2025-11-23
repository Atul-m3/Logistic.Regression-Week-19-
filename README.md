# Logistic Regression – Scratch vs Sklearn

## Overview
This project explores logistic regression from scratch using NumPy and compares it with sklearn’s implementation. It includes training, prediction, visualization, and performance analysis.

## Files
- `manual_logistic_regression.py`: Logistic regression built from scratch  
- `sklearn_logistic_regression.py`: Sklearn implementation and comparison  
- `visualization_analysis.py`: Decision boundaries, confusion matrices, and classification reports  
- `visuals/`: Saved plots (sigmoid curve, decision boundaries)  
- `model_evaluation.ipynb`: Evaluation notebook with metrics and visualizations

## Reflection
Building the model manually helped me understand how gradient descent works and how predictions are formed. Sklearn is faster and optimized, but coding it myself made me think like an engineer, not just a tool user.

## Dataset
Simple binary classification using student exam scores and study hours.

## Requirements
- Python  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## Model Evaluation Summary

This section summarizes how I evaluated my logistic regression model using multiple metrics and visualizations.

### Metrics Explained
- **Accuracy**: Measures how often the model predicts correctly overall.
- **Precision**: Out of all predicted positives, how many were actually correct.
- **Recall**: Out of all actual positives, how many did the model correctly identify.
- **F1 Score**: Combines precision and recall into one balanced metric.
- **AUC (Area Under Curve)**: Measures how well the model separates the classes.

### Visualizations
- **Confusion Matrix**: Shows correct and incorrect predictions for each class.
- **ROC Curve**: Displays the trade-off between true positive rate and false positive rate.

### Class Imbalance Test
I tested the model on an imbalanced dataset (80% class 0, 20% class 1).  
Even though accuracy was 80%, precision, recall, and F1 Score dropped to 0.  
This shows why accuracy alone can be misleading.

### Screenshots / Results
All metric outputs and plots are included in:  
**`model_evaluation.ipynb`**

# Logistic Regression – Scratch vs Sklearn

## Overview
This project explores logistic regression from scratch using NumPy and compares it with sklearn’s implementation. It includes training, prediction, visualization, and performance analysis. The goal is to understand how logistic regression works internally and how regularization improves model generalization.

## Files
- `manual_logistic_regression.py`: Logistic regression built from scratch  
- `sklearn_logistic_regression.py`: Sklearn implementation and comparison  
- `visualization_analysis.py`: Decision boundaries, confusion matrices, and classification reports  
- `regularization_experiments.ipynb`: Regularization experiments with L1, L2, and coefficient shrinkage  
- `visuals/`: Saved plots (sigmoid curve, decision boundaries, shrinkage plots)

## Reflection
Building the model manually helped me understand how gradient descent works and how predictions are formed. Sklearn is faster and optimized, but coding it myself made me think like an engineer, not just a tool user. Regularization taught me how to control model complexity and improve generalization a key skill for building reliable AI systems.

## Dataset
Simple binary classification using student exam scores and study hours. Target variable: whether the student passed (1) or not (0).

## Requirements
- Python  
- NumPy  
- Matplotlib  
- Scikit-learn  

## Preventing Overfitting

This section focuses on improving model generalization by applying regularization techniques to logistic regression.

### What I Implemented
- Trained three logistic regression models:
  - **No Regularization**: Simulated using a very high C value to minimize penalty.
  - **L1 Regularization**: Encourages sparsity by driving some coefficients to zero.
  - **L2 Regularization**: Smoothly shrinks all coefficients to reduce complexity.
- Evaluated each model using:
  - **Accuracy**
  - **F1 Score**
  - **Coefficients**
- Visualized how coefficients shrink as regularization strength increases (`C` values from 0.01 to 100).

### Key Insights
- **Overfitting** occurs when a model memorizes training data and fails to generalize.
- **Regularization** helps control model complexity:
  - **L1 (Lasso)** is ideal for feature selection.
  - **L2 (Ridge)** is better for smoothing and generalization.
- As regularization increases (lower C), weights shrink and the model becomes simpler.
- A simpler model may lose a bit of accuracy but becomes more reliable on unseen data.

## License
This project is open-source and free to use for educational purposes.

## Optimizing Hyperparameters

To make my model more reliable, I used **k-fold cross-validation** and **GridSearchCV**.  
- CV gave mean accuracy = 0.90 ± 0.20 and F1 = 0.93 ± 0.13, showing consistent performance.  
- GridSearchCV found the best parameter: **C = 0.01**, with F1 ≈ 0.93 and accuracy = 1.0.  
- The validation curve (C vs F1) showed how regularization strength impacts generalization.

**Why CV matters:** Cross-validation improves trust in results because it tests the model across multiple splits, reducing randomness and giving a more realistic estimate of performance.
