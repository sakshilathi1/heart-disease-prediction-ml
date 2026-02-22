# Heart Disease Prediction using Machine Learning

## Project Overview
This project builds and evaluates six supervised machine learning 
classifiers to predict heart disease using the UCI Heart Disease 
dataset (270 patients, 13 clinical features).

Built as part of EEE 591 - Python for rapid Engineering Solutions 
Arizona State University | Spring 2026

---

## Dataset
- 270 patient records
- 13 medical features (age, chest pain type, max heart rate, etc.)
- Target: absence (1) or presence (2) of heart disease
- Balanced: 55.6% no disease, 44.4% disease

---

## Problem 1: Data Analysis (problem1.py)
- Full correlation matrix and heatmap
- Point-biserial correlation with p-value significance testing
- Distribution analysis (skewness and kurtosis for all features)
- Class balance analysis
- Outlier detection using IQR method
- Pair plot visualization

**Key Finding:** thal (r=0.525), nmvcf (r=0.455), and eia (r=0.419) 
are the strongest predictors of heart disease.

---

## Problem 2: ML Classifiers (problem2.py)
Six classifiers trained and evaluated on 80/20 train-test split 
with StandardScaler normalization:

| Method | Test Accuracy | AUC |
|---|---|---|
| Support Vector Machines | **0.94** | **0.978** |
| Random Forest | 0.93 | 0.967 |
| Logistic Regression | 0.91 | 0.968 |
| K-Nearest Neighbor (K=11) | 0.91 | 0.965 |
| Perceptron | 0.85 | N/A |
| Decision Tree | 0.78 | 0.773 |

**Best Model:** SVM with RBF kernel (Test Accuracy: 0.94, AUC: 0.978)

**Cross-Validation:** KNN achieved best CV accuracy of 0.8444 
with lowest variance (+/-0.057), indicating most reliable 
generalization.

---

## Visualizations
| Plot | Description |
|---|---|
| correlation_heatmap.png | Full 14x14 correlation matrix |
| feature_importance_stats.png | Point-biserial correlation bar chart |
| confusion_matrices.png | All 6 classifiers on test set |
| roc_curves.png | ROC curves with AUC values |
| learning_curves.png | Training vs CV score for top 2 models |
| knn_k_selection.png | K=1 to 30 CV accuracy plot |
| pairplot.png | Feature pair plots colored by diagnosis |

---

## Tech Stack
- Python 3
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- SciPy

---

## How to Run
```bash
# Clone the repo
git clone https://github.com/sakshilathi1/heart-disease-prediction-ml.git
cd heart-disease-prediction-ml

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn scipy

# Run data analysis
python3 problem1.py

# Run ML classifiers
python3 problem2.py
```

## Results
All outputs are saved to the `outputs/` folder automatically.

---

## Author
Sakshi | MS Robotics and Autonomous Systems (AI concentration)  
Arizona State University
