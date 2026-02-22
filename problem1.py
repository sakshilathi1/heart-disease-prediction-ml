import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

#  load data
csv_path = os.path.join(script_dir, 'heart1.csv')
df = pd.read_csv(csv_path)

#  basic dataset info 
print("=" * 70)
print("DATASET SHAPE")
print("=" * 70)
print(df.shape)
print()

print("=" * 70)
print("FIRST 10 ROWS")
print("=" * 70)
print(df.head(10))
print()

print("=" * 70)
print("DATA TYPES")
print("=" * 70)
print(df.dtypes)
print()

print("=" * 70)
print("DESCRIPTIVE STATISTICS (ALL 14 COLUMNS)")
print("=" * 70)
print(df.describe())
print()

print("=" * 70)
print("MISSING VALUES CHECK")
print("=" * 70)
print(df.isnull().sum())  # no nulls in this dataset
print()

#  correlation matrix  
corr_matrix = df.corr()

print("=" * 70)
print("FULL CORRELATION MATRIX")
print("=" * 70)
# widen display so the matrix doesn't wrap around
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)
print(corr_matrix.to_string())
print()

#  each feature's correlation with target 
target_corr = corr_matrix['a1p2'].drop('a1p2')
# sort by absolute value so strongest show first
target_corr_sorted = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)

print("=" * 70)
print("CORRELATION OF EACH FEATURE WITH TARGET (a1p2)")
print("Sorted by absolute correlation value (descending)")
print("=" * 70)
for feat, val in target_corr_sorted.items():
    print(f"  {feat:<10s}  correlation with a1p2: {val:+.4f}  (|r| = {abs(val):.4f})")
print()

#  top 5 inter-feature pairs 
print("=" * 70)
print("TOP 5 FEATURE PAIRS MOST CORRELATED WITH EACH OTHER")
print("(Excluding self-correlation and duplicates)")
print("=" * 70)

# upper triangle only, drop a1p2 so we only look at feature-feature
feature_corr = corr_matrix.drop('a1p2', axis=0).drop('a1p2', axis=1)
upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
pairs = []
for col in upper.columns:
    for idx in upper.index:
        val = upper.loc[idx, col]
        if pd.notna(val):
            pairs.append((idx, col, val, abs(val)))

pairs_sorted = sorted(pairs, key=lambda x: x[3], reverse=True)

for i, (f1, f2, corr_val, abs_val) in enumerate(pairs_sorted[:5]):
    print(f"  {i+1}. {f1:<8s} <-> {f2:<8s}  r = {corr_val:+.4f}  (|r| = {abs_val:.4f})")
print()

#  top 5 features vs target 
print("=" * 70)
print("TOP 5 FEATURES MOST CORRELATED WITH a1p2 (TARGET)")
print("=" * 70)
for i, (feat, val) in enumerate(list(target_corr_sorted.items())[:5]):
    print(f"  {i+1}. {feat:<10s}  r = {val:+.4f}  (|r| = {abs(val):.4f})")
print()

#  heatmap 
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix - Heart Disease Dataset (EEE 591 Project 1)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/correlation_heatmap.png")
print()

#  pair plot (takes a while with 14 cols) 
print("Generating pair plot (this may take a moment)...")
try:
    g = sns.pairplot(df, hue='a1p2', diag_kind='hist',
                     plot_kws={'alpha': 0.5, 's': 15})
    g.figure.suptitle('Pair Plot - Heart Disease Dataset',
                      y=1.01, fontsize=16, fontweight='bold')
    g.savefig(os.path.join(output_dir, 'pairplot.png'),
              dpi=100, bbox_inches='tight')
    plt.close('all')
    print("Saved: outputs/pairplot.png")
except Exception as e:
    print(f"Note: Pairplot could not be generated: {e}")
    print("Skipping pairplot and continuing with analysis.")
    plt.close('all')
print()

# store top correlations for the management summary later
top5_target = list(target_corr_sorted.items())[:5]
top5_pairs = pairs_sorted[:5]
feat1_name, feat1_val = top5_target[0]
feat2_name, feat2_val = top5_target[1]
feat3_name, feat3_val = top5_target[2]
feat4_name, feat4_val = top5_target[3]
feat5_name, feat5_val = top5_target[4]
pair1_f1, pair1_f2, pair1_corr, _ = top5_pairs[0]
pair2_f1, pair2_f2, pair2_corr, _ = top5_pairs[1]

# as per instructions need to use all 13 features so not dropping any
features = [col for col in df.columns if col != 'a1p2']

# skewness and kurtosis 
print("=" * 70)
print("DISTRIBUTION ANALYSIS - Skewness and Kurtosis")
print("=" * 70)
print(f"  {'Feature':<10s}  {'Skewness':>10s}  {'Kurtosis':>10s}  {'Distribution':<20s}")
print("-" * 60)

skew_data = {}
for feat in features:
    skew = df[feat].skew()
    kurt = df[feat].kurtosis()
    # 0.5 threshold for classifying skew direction
    if abs(skew) < 0.5:
        dist_type = "Approx. Normal"
    elif skew > 0:
        dist_type = "Right-Skewed"
    else:
        dist_type = "Left-Skewed"
    skew_data[feat] = {'skewness': skew, 'kurtosis': kurt, 'type': dist_type}
    print(f"  {feat:<10s}  {skew:>10.4f}  {kurt:>10.4f}  {dist_type:<20s}")
print()

# histograms split by class with kde overlay
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes_flat = axes.flatten()
colors_cls = {1: '#3498db', 2: '#e74c3c'}
labels_cls = {1: 'No Disease (1)', 2: 'Disease (2)'}

for i, feat in enumerate(features):
    ax = axes_flat[i]
    for cls in [1, 2]:
        subset = df[df['a1p2'] == cls][feat]
        ax.hist(subset, bins=20, alpha=0.5, color=colors_cls[cls],
                label=labels_cls[cls], density=True, edgecolor='white')
        # kde crashes on binary/low-unique features so skip those
        if df[feat].nunique() > 10:
            try:
                subset.plot.kde(ax=ax, color=colors_cls[cls], linewidth=1.5)
            except Exception:
                pass
    ax.set_title(f'{feat} (skew={skew_data[feat]["skewness"]:.2f})',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

# hide unused subplot slots (13 features, 16 grid spots)
for j in range(len(features), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle('Feature Distributions by Heart Disease Status',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/feature_distributions.png")
print()

# class balance 
print("=" * 70)
print("CLASS BALANCE ANALYSIS")
print("=" * 70)
class_counts = df['a1p2'].value_counts().sort_index()
total = len(df)
for cls, count in class_counts.items():
    pct = count / total * 100
    label = "No Heart Disease" if cls == 1 else "Heart Disease"
    print(f"  Class {cls} ({label}): {count} samples ({pct:.1f}%)")

minority_pct = class_counts.min() / total * 100
# 40% threshold from class - below that is considered imbalanced
if minority_pct < 40:
    balance_status = "IMBALANCED"
    print(f"  Dataset Status: {balance_status} (minority class = {minority_pct:.1f}% < 40%)")
else:
    balance_status = "BALANCED"
    print(f"  Dataset Status: {balance_status} (minority class = {minority_pct:.1f}% >= 40%)")
print()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(['Class 1\n(No Disease)', 'Class 2\n(Disease)'],
              [class_counts.get(1, 0), class_counts.get(2, 0)],
              color=['#3498db', '#e74c3c'], edgecolor='white', width=0.5)
for bar, count in zip(bars, [class_counts.get(1, 0), class_counts.get(2, 0)]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(count), ha='center', va='bottom', fontweight='bold', fontsize=14)
ax.set_title('Class Distribution - Heart Disease Dataset', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=12)
ax.set_ylim(0, max(class_counts) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/class_distribution.png")
print()

# point-biserial correlation 
# picked this over plain pearson bc it handles binary targets + gives p-values
print("=" * 70)
print("FEATURE IMPORTANCE - Point-Biserial Correlation with a1p2")
print("(Statistical significance test: p < 0.05)")
print("=" * 70)

binary_target = (df['a1p2'] - 1).values  # remap 1/2 -> 0/1 for the test

pb_results = {}
for feat in features:
    corr_pb, p_val = stats.pointbiserialr(binary_target, df[feat].values)
    pb_results[feat] = {'corr': corr_pb, 'pval': p_val, 'sig': p_val < 0.05}

pb_sorted = sorted(pb_results.items(), key=lambda x: abs(x[1]['corr']), reverse=True)

print(f"  {'Feature':<10s}  {'Correlation':>12s}  {'p-value':>12s}  {'Significant?':<14s}")
print("-" * 55)
for feat, vals in pb_sorted:
    sig_str = "Yes (p<0.05)" if vals['sig'] else "No"
    print(f"  {feat:<10s}  {vals['corr']:>+12.4f}  {vals['pval']:>12.2e}  {sig_str:<14s}")
print()

sig_features = [f for f, v in pb_sorted if v['sig']]
nonsig_features = [f for f, v in pb_sorted if not v['sig']]
# not sure why sc and fbs have such low correlation but keeping them
print(f"  Significant predictors ({len(sig_features)}): {', '.join(sig_features)}")
print(f"  Non-significant ({len(nonsig_features)}): {', '.join(nonsig_features)}")
print()

fig, ax = plt.subplots(figsize=(10, 7))
feat_names_pb = [f for f, _ in pb_sorted]
feat_corrs_pb = [abs(v['corr']) for _, v in pb_sorted]
feat_colors_pb = ['#2ecc71' if v['sig'] else '#e74c3c' for _, v in pb_sorted]

ax.barh(range(len(feat_names_pb)), feat_corrs_pb, color=feat_colors_pb, edgecolor='white')
ax.set_yticks(range(len(feat_names_pb)))
ax.set_yticklabels(feat_names_pb, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('|Point-Biserial Correlation|', fontsize=12)
ax.set_title('Feature Importance (Point-Biserial Correlation with a1p2)\n'
             'Green = Significant (p<0.05), Red = Not Significant',
             fontsize=13, fontweight='bold')
for i, corr_val in enumerate(feat_corrs_pb):
    ax.text(corr_val + 0.005, i, f'{corr_val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_stats.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/feature_importance_stats.png")
print()

# outlier detection (IQR) 
print("=" * 70)
print("OUTLIER DETECTION (IQR Method: Q1-1.5*IQR to Q3+1.5*IQR)")
print("=" * 70)
print(f"  {'Feature':<10s}  {'# Outliers':>12s}  {'% Outliers':>12s}")
print("-" * 40)

outlier_data = {}
for feat in features:
    Q1 = df[feat].quantile(0.25)
    Q3 = df[feat].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    n_outliers = len(df[(df[feat] < lower_bound) | (df[feat] > upper_bound)])
    pct = n_outliers / len(df) * 100
    outlier_data[feat] = {'count': n_outliers, 'pct': pct}
    print(f"  {feat:<10s}  {n_outliers:>12d}  {pct:>11.1f}%")

total_outlier_features = sum(1 for v in outlier_data.values() if v['count'] > 0)
print(f"\n  Features with outliers: {total_outlier_features}/{len(features)}")
print()

fig, axes = plt.subplots(4, 4, figsize=(18, 12))
axes_flat = axes.flatten()
for i, feat in enumerate(features):
    ax = axes_flat[i]
    bp = ax.boxplot(df[feat], patch_artist=True, vert=True,
                    flierprops=dict(marker='o', markerfacecolor='red',
                                    markersize=5, alpha=0.6),
                    boxprops=dict(facecolor='#3498db', alpha=0.6),
                    medianprops=dict(color='black', linewidth=2))
    ax.set_title(f'{feat} ({outlier_data[feat]["count"]} outliers)',
                 fontweight='bold', fontsize=10)
    ax.tick_params(labelsize=8)

for j in range(len(features), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle('Boxplots with Outlier Detection (Red = Outliers)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/boxplots.png")
print()

# management summary 
# group features by correlation strength
strong = [f for f, v in pb_sorted if abs(v['corr']) >= 0.4]
moderate = [f for f, v in pb_sorted if 0.2 <= abs(v['corr']) < 0.4]
weak = [f for f, v in pb_sorted if abs(v['corr']) < 0.2]
high_outlier = [f for f, v in outlier_data.items() if v['pct'] > 5]

print("=" * 70)
print("AMAPE MANAGEMENT SUMMARY - Problem 1 Data Analysis")
print("=" * 70)
print(f"""
To the Management of AMAPE (Acme Medical Analysis and Prediction Enterprises):

After analyzing the heart disease dataset with {len(df)} patient records and 13 features, here is what I found. The features most strongly linked to heart disease were {feat1_name} (r = {feat1_val:+.4f}), {feat2_name} (r = {feat2_val:+.4f}), {feat3_name} (r = {feat3_val:+.4f}), {feat4_name} (r = {feat4_val:+.4f}), and {feat5_name} (r = {feat5_val:+.4f}). {feat1_name} being the strongest predictor makes sense since the thallium stress test is a direct measure of heart function. I also noticed that {pair1_f1} and {pair1_f2} are highly correlated with each other (r = {pair1_corr:+.4f}), and {pair2_f1} and {pair2_f2} are negatively correlated (r = {pair2_corr:+.4f}), meaning older patients tend to have lower maximum heart rates.

The dataset is fairly balanced with {class_counts.get(1, 0)} patients without heart disease and {class_counts.get(2, 0)} with it ({minority_pct:.1f}% minority class), so we {'should not need any special balancing techniques' if balance_status == 'BALANCED' else 'may need to use resampling or class weighting to handle the imbalance'}. Out of 13 features, {len(sig_features)} turned out to be statistically significant predictors based on point-biserial correlation testing at p < 0.05.
{'Only ' + ', '.join(nonsig_features) + ' were not significant, but I kept all features as required.' if nonsig_features else 'All features were significant.'}

I also checked for outliers using the IQR method and found that {total_outlier_features} out of 13 features have outliers{', especially ' + ', '.join(f'{f} ({outlier_data[f]["pct"]:.1f}%)' for f in high_outlier[:3]) if high_outlier else ''}. This is something to keep in mind since outliers can hurt distance-based models like KNN, though tree-based methods handle them fine.

Overall, I am pretty confident that {', '.join(strong)} will be the strongest predictors for our models.The features {', '.join(moderate) if moderate else 'at a moderate level'} should also help, while {', '.join(weak) if weak else 'no features'} have weaker signal but might still add some value in ensemble methods. Based on all of this, I think machine learning models should be able to predict heart disease pretty well, probably in the range of 0.80 to 0.94 accuracy, with SVM and Random Forest likely doing the best.
""")
print("=" * 70)
print("End of Problem 1 Analysis")
print("=" * 70)
