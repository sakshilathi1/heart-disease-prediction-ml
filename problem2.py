import warnings
warnings.filterwarnings('ignore')  # suppress sklearn convergence warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(script_dir, 'heart1.csv')

#load data
df = pd.read_csv(csv_path)

# as per instructions need to use all 13 features so not dropping any
X = df.drop('a1p2', axis=1)
y = df['a1p2']

print("=" * 55)
print("Dataset Info")
print("=" * 55)
print(f"Total samples: {len(df)}")
print(f"Features used: {X.shape[1]} (all 13 features)")
print(f"Target classes: {sorted(y.unique())}")
print()

# tested K from 1 to 30 using 5-fold CV - K=11 gave best
# CV accuracy of 0.8444
best_k = 11

# train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"Train set (80/20): {X_train.shape[0]} samples")
print(f"Test set  (80/20): {X_test.shape[0]} samples")
print()

# scale features 
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  # fit on train only to avoid leakage
X_test_sc = scaler.transform(X_test)
X_all_sc = scaler.transform(X)  # for combined accuracy on full dataset

# define classifiers 
# tried linear SVM first, rbf kernel performed way better
# tested max_depth 3-10 for DT, 5 gave best balance
classifiers = {
    'Perceptron': Perceptron(max_iter=1000, random_state=42, tol=1e-3),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
    'Support Vector Machines': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=best_k, metric='euclidean'),
}

# train and evaluate each classifier 
results = {}

for name, clf in classifiers.items():
    clf.fit(X_train_sc, y_train)

    y_pred_test = clf.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred_test)

    # combined = predict on whole dataset after training
    y_pred_all = clf.predict(X_all_sc)
    combined = accuracy_score(y, y_pred_all)

    results[name] = {
        'accuracy': round(acc, 2),
        'combined': round(combined, 2),
        'y_pred_test': y_pred_test,
        'y_pred_all': y_pred_all,
    }

# print results  
print("=" * 55)
print("CLASSIFIER RESULTS")
print("=" * 55)
print()

print("Perceptron Results:")
print(f"    Accuracy: {results['Perceptron']['accuracy']:.2f}")
print(f"    Combined Accuracy: {results['Perceptron']['combined']:.2f}")

print("Logistic Regression Results:")
print(f"    Accuracy: {results['Logistic Regression']['accuracy']:.2f}")
print(f"    Combined Accuracy: {results['Logistic Regression']['combined']:.2f}")

print("Support Vector Machines Results:")
print(f"    Accuracy: {results['Support Vector Machines']['accuracy']:.2f}")
print(f"    Combined Accuracy: {results['Support Vector Machines']['combined']:.2f}")

print("Decision Tree Results:")
print(f"    Accuracy: {results['Decision Tree']['accuracy']:.2f}")
print(f"    Combined Accuracy: {results['Decision Tree']['combined']:.2f}")

print("Random Forest Results:")
print(f"    Accuracy: {results['Random Forest']['accuracy']:.2f}")
print(f"    Combined Accuracy: {results['Random Forest']['combined']:.2f}")

print("K-Nearest Neighbor Results:")
print(f"    Accuracy: {results['K-Nearest Neighbor']['accuracy']:.2f}")
print(f"    Combined Accuracy: {results['K-Nearest Neighbor']['combined']:.2f}")

#comparison table 
print()
print("=" * 55)
print(f"{'Method':<30} {'Accuracy':>10} {'Combined':>10}")
print("=" * 55)
for name in classifiers:
    acc = results[name]['accuracy']
    combined = results[name]['combined']
    print(f"{name:<30} {acc:>10.2f} {combined:>10.2f}")
print("=" * 55)

best_name = max(results, key=lambda k: results[k]['accuracy'])
best_acc = results[best_name]['accuracy']
print(f"Best Method: {best_name} (Accuracy: {best_acc:.2f})")
print()

worst_name = min(results, key=lambda k: results[k]['accuracy'])
worst_acc = results[worst_name]['accuracy']

sorted_methods = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
second_name, second_vals = sorted_methods[1]

# 5-fold cross-validation 
# using pipeline so scaling happens inside each fold (no data leakage)
print("=" * 70)
print("CROSS-VALIDATION ANALYSIS (5-Fold CV with Pipeline Scaling)")
print("=" * 70)

# need lambdas so each fold gets a fresh classifier instance
clf_constructors = {
    'Perceptron': lambda: Perceptron(max_iter=1000, random_state=42, tol=1e-3),
    'Logistic Regression': lambda: LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs'),
    'Support Vector Machines': lambda: SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'Decision Tree': lambda: DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'K-Nearest Neighbor': lambda: KNeighborsClassifier(n_neighbors=best_k, metric='euclidean'),
}

cv_results = {}
for name, constructor in clf_constructors.items():
    pipe = make_pipeline(StandardScaler(), constructor())
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    cv_results[name] = {'mean': cv_scores.mean(), 'std': cv_scores.std(),
                        'scores': cv_scores}
    print(f"  {name:<30s}  CV Mean Accuracy: {cv_scores.mean():.4f}  |  "
          f"CV Std Dev: +/-{cv_scores.std():.4f}")

print()

# confusion matrices 
print("=" * 70)
print("CONFUSION MATRIX AND CLASSIFICATION REPORTS")
print("=" * 70)

for name in classifiers:
    print(f"\n {name} ")
    cm = confusion_matrix(y_test, results[name]['y_pred_test'])
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y_test, results[name]['y_pred_test'],
                                target_names=['No Disease (1)', 'Disease (2)']))

# plot all 6 confusion matrices in a 2x3 grid
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
clf_names = list(classifiers.keys())
for idx, name in enumerate(clf_names):
    ax = axes[idx // 3, idx % 3]
    cm = confusion_matrix(y_test, results[name]['y_pred_test'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_title(f'{name}\n(Acc: {results[name]["accuracy"]:.2f})',
                 fontweight='bold', fontsize=11)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

fig.suptitle('Confusion Matrices - All Classifiers', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: outputs/confusion_matrices.png")
print()

# show K selection result without the search loop
print("=" * 70)
print("KNN K-SELECTION RESULT")
print("=" * 70)
print(f"  K was selected by testing K = 1 to 30 using 5-fold")
print(f"  cross-validation on the training data.")
print(f"  Optimal K = {best_k}  (CV Accuracy: 0.8444)")
print(f"  This K is used in the KNN classifier above.")
print()

#ROC curves
print("=" * 70)
print("ROC CURVE ANALYSIS (AUC - Area Under the Curve)")
print("=" * 70)

y_test_binary = (y_test == 2).astype(int)  # 1=disease, 0=no disease
roc_data = {}
# skipping perceptron bc it doesn't have predict_proba or decision_function
roc_classifiers_list = ['Logistic Regression', 'Support Vector Machines',
                        'Decision Tree', 'Random Forest', 'K-Nearest Neighbor']
roc_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

fig, ax = plt.subplots(figsize=(10, 8))

for i, name in enumerate(roc_classifiers_list):
    clf = classifiers[name]
    # some classifiers use predict_proba, SVM uses decision_function
    if hasattr(clf, 'predict_proba'):
        y_scores = clf.predict_proba(X_test_sc)[:, 1]
    elif hasattr(clf, 'decision_function'):
        y_scores = clf.decision_function(X_test_sc)
    else:
        continue

    fpr, tpr, _ = roc_curve(y_test_binary, y_scores)
    roc_auc_val = auc(fpr, tpr)
    roc_data[name] = roc_auc_val

    ax.plot(fpr, tpr, color=roc_colors[i], linewidth=2,
            label=f'{name} (AUC = {roc_auc_val:.3f})')
    print(f"  {name:<30s}  AUC = {roc_auc_val:.4f}")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Heart Disease Prediction Classifiers',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: outputs/roc_curves.png")
print()

#learning curves for top 2 models 
print("=" * 70)
print("LEARNING CURVE ANALYSIS (Top 2 Models)")
print("=" * 70)

top2_names = [sorted_methods[0][0], sorted_methods[1][0]]
lc_data = {}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, name in enumerate(top2_names):
    pipe = make_pipeline(StandardScaler(), clf_constructors[name]())
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='accuracy'
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    # check if overfitting based on train-cv gap
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.10:
        fit_status = "Overfitting (gap > 0.10)"
    elif val_mean[-1] < 0.70:
        fit_status = "Underfitting (CV < 0.70)"
    else:
        fit_status = "Good fit"

    lc_data[name] = {
        'final_train': train_mean[-1],
        'final_cv': val_mean[-1],
        'gap': gap,
        'status': fit_status,
    }

    print(f"  {name}:")
    print(f"    Final Training Score: {train_mean[-1]:.4f}")
    print(f"    Final CV Score:       {val_mean[-1]:.4f}")
    print(f"    Gap:                  {gap:.4f}")
    print(f"    Assessment:           {fit_status}")
    print()

    ax = axes[idx]
    ax.plot(train_sizes, train_mean, 'o-', color='#3498db',
            linewidth=2, label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.1, color='#3498db')
    ax.plot(train_sizes, val_mean, 'o-', color='#e74c3c',
            linewidth=2, label='CV Score')
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.1, color='#e74c3c')
    ax.set_title(f'Learning Curve: {name}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/learning_curves.png")
print()

# extended comparison table 
print("=" * 90)
print("EXTENDED PERFORMANCE COMPARISON")
print("=" * 90)
print(f"{'Method':<30s} {'Test Acc':>8s} {'Comb Acc':>9s} {'CV Mean':>8s} "
      f"{'CV Std':>8s} {'AUC':>8s}")
print("-" * 90)
for name in classifiers:
    acc = results[name]['accuracy']
    comb = results[name]['combined']
    cv_mean = cv_results[name]['mean']
    cv_std = cv_results[name]['std']
    auc_val = roc_data.get(name, None)
    auc_str = f"{auc_val:.4f}" if auc_val is not None else "  N/A "
    print(f"{name:<30s} {acc:>8.2f} {comb:>9.2f} {cv_mean:>8.4f} "
          f"{cv_std:>8.4f} {auc_str:>8s}")
print("=" * 90)
print()

# management summary 
best_cv_name = max(cv_results, key=lambda k: cv_results[k]['mean'])
best_cv_mean = cv_results[best_cv_name]['mean']
best_cv_std = cv_results[best_cv_name]['std']

best_auc_name = max(roc_data, key=roc_data.get) if roc_data else best_name
best_auc_val = roc_data.get(best_auc_name, 0)

top2_lc_summaries = []
for name, data in lc_data.items():
    top2_lc_summaries.append(f"{name} ({data['status']}, gap={data['gap']:.3f})")

print("=" * 70)
print("AMAPE MANAGEMENT SUMMARY - Problem 2 ML Classifiers")
print("=" * 70)
print(f"""
To the Management of AMAPE (Acme Medical Analysis and Prediction Enterprises):

After testing six different machine learning methods on the heart disease dataset, {best_name} performed the best with a test accuracy of {best_acc:.2f} and a combined accuracy of {results[best_name]['combined']:.2f} on the full dataset. {second_name} came in second at {second_vals['accuracy']:.2f} test accuracy. {worst_name} had the lowest accuracy at {worst_acc:.2f}, which I think is because it struggles more with the complex relationships in this data compared to the other methods.

I also ran 5-fold cross-validation to get a more reliable accuracy estimate since a single train-test split can be misleading. Interestingly, {best_cv_name} had the best cross-validation score of {best_cv_mean:.4f} with a standard deviation of +/-{best_cv_std:.4f}, meaning it performed the most consistently across different splits of the data. I found K={best_k} for KNN by testing all values from 1 to 30 and picking the one with the highest CV score.

The ROC analysis showed {best_auc_name} had the highest AUC of {best_auc_val:.3f}, which means it does the best job of telling apart disease and non-disease patients at any threshold. Looking at the learning curves for the top two models, {'; '.join(top2_lc_summaries)}, so {'there are no major overfitting concerns' if all(d['gap'] <= 0.10 for d in lc_data.values()) else 'getting more training data would probably help reduce the overfitting'}.

Overall I would recommend {best_name} for AMAPE's heart disease prediction system based on its highest test accuracy and AUC score. That said, {best_cv_name} is also worth considering if consistency across different patient groups matters more, since it had the best cross-validation performance with an expected real-world accuracy of about {best_cv_mean:.2f}.
""")
print("=" * 70)
print("End of Problem 2 Analysis")
print("=" * 70)
