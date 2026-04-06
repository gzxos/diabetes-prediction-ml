import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =====================
# FIXED FEATURES (SAMA SEPERTI train.py)
# =====================
SELECTED_FEATURES = [
    'HighBP', 'HighChol', 'BMI', 'AnyHealthcare', 
    'GenHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Income'
]

# =====================
# LOAD DATA
# =====================
print("="*60)
print("STRATIFIED K-FOLD CROSS VALIDATION - LOGISTIC REGRESSION")
print("="*60)

print("\nLoading dataset...")
df = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")
df = df.rename(columns={'Diabetes_binary': 'diabetes'})
print(f"Dataset loaded: {df.shape}")

# =====================
# FILTER POPULATION (USIA PRODUKTIF 18-44 tahun)
# =====================
print("\nFiltering age 18-44 (fokus usia produktif)...")
print(f"Before filter: {df.shape}")
df = df[df['Age'] <= 5].copy()  # Age 1-5 = 18-44 tahun
print(f"After filter: {df.shape}")

# =====================
# DROP DUPLICATES
# =====================
print(f"\nDuplicates before: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"After drop duplicates: {df.shape}")

# =====================
# CHECK DATA BALANCE
# =====================
print("\nTarget distribution:")
print(df['diabetes'].value_counts())
print("\nPercentage:")
print(df['diabetes'].value_counts(normalize=True) * 100)

# =====================
# SELECT FIXED FEATURES (SAMA SEPERTI train.py)
# =====================
print("\n" + "="*60)
print(f"USING FIXED {len(SELECTED_FEATURES)} FEATURES")
print("="*60)
for i, feat in enumerate(SELECTED_FEATURES, 1):
    print(f"  {i}. {feat}")

y = df['diabetes']
X = df[SELECTED_FEATURES].copy()

print(f"\nTotal features: {X.shape[1]}")

# =====================
# STRATIFIED K-FOLD CV SETUP
# =====================
n_splits = 5

print(f"\n{'='*60}")
print(f"PERFORMING {n_splits}-FOLD STRATIFIED CROSS VALIDATION")
print(f"MODEL: Logistic Regression (max_iter=1000)")
print(f"FEATURES: {len(SELECTED_FEATURES)} FIXED features")
print(f"{'='*60}")

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Storage for metrics
cv_results = {
    'fold': [],
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'roc_auc': [],
    'tn': [],
    'fp': [],
    'fn': [],
    'tp': []
}

# =====================
# CROSS VALIDATION LOOP
# =====================
fold_num = 1

for train_idx, test_idx in skf.split(X, y):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_num}/{n_splits}")
    print(f"{'='*60}")
    
    # Split data
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Train set: {X_train_fold.shape}, Test set: {X_test_fold.shape}")
    
    # =====================
    # SCALING (SAMA SEPERTI train.py)
    # =====================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled = scaler.transform(X_test_fold)
    
    print(f"Features scaled: {X_train_scaled.shape[1]}")
    
    # =====================
    # HANDLE IMBALANCED DATA WITH SMOTE (SAMA SEPERTI train.py)
    # =====================
    print(f"\nBefore SMOTE - Class 0: {(y_train_fold == 0).sum()}, Class 1: {(y_train_fold == 1).sum()}")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train_fold)
    
    print(f"After SMOTE  - Class 0: {(y_train_balanced == 0).sum()}, Class 1: {(y_train_balanced == 1).sum()}")
    
    # =====================
    # TRAIN MODEL (LOGISTIC REGRESSION - SAMA SEPERTI train.py)
    # =====================
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_balanced, y_train_balanced)
    
    # =====================
    # PREDICT ON TEST FOLD
    # =====================
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # =====================
    # CALCULATE METRICS
    # =====================
    acc = accuracy_score(y_test_fold, y_pred)
    prec = precision_score(y_test_fold, y_pred, zero_division=0)
    rec = recall_score(y_test_fold, y_pred, zero_division=0)
    f1 = f1_score(y_test_fold, y_pred, zero_division=0)
    auc = roc_auc_score(y_test_fold, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_fold, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Store results
    cv_results['fold'].append(fold_num)
    cv_results['accuracy'].append(acc)
    cv_results['precision'].append(prec)
    cv_results['recall'].append(rec)
    cv_results['f1_score'].append(f1)
    cv_results['roc_auc'].append(auc)
    cv_results['tn'].append(tn)
    cv_results['fp'].append(fp)
    cv_results['fn'].append(fn)
    cv_results['tp'].append(tp)
    
    # Print fold results
    print(f"\nFold {fold_num} Results:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f} ⭐")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {tn:>5}  FP: {fp:>5}")
    print(f"    FN: {fn:>5}  TP: {tp:>5}")
    
    fold_num += 1

# =====================
# CV SUMMARY STATISTICS
# =====================
print(f"\n{'='*60}")
print("CROSS VALIDATION SUMMARY")
print(f"{'='*60}")

cv_df = pd.DataFrame(cv_results)

# Calculate mean and std
summary_stats = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Mean': [
        np.mean(cv_results['accuracy']),
        np.mean(cv_results['precision']),
        np.mean(cv_results['recall']),
        np.mean(cv_results['f1_score']),
        np.mean(cv_results['roc_auc'])
    ],
    'Std': [
        np.std(cv_results['accuracy']),
        np.std(cv_results['precision']),
        np.std(cv_results['recall']),
        np.std(cv_results['f1_score']),
        np.std(cv_results['roc_auc'])
    ],
    'Min': [
        np.min(cv_results['accuracy']),
        np.min(cv_results['precision']),
        np.min(cv_results['recall']),
        np.min(cv_results['f1_score']),
        np.min(cv_results['roc_auc'])
    ],
    'Max': [
        np.max(cv_results['accuracy']),
        np.max(cv_results['precision']),
        np.max(cv_results['recall']),
        np.max(cv_results['f1_score']),
        np.max(cv_results['roc_auc'])
    ]
}

summary_df = pd.DataFrame(summary_stats)

print("\n" + summary_df.to_string(index=False))

# Detailed format
print(f"\n{'='*60}")
print("DETAILED METRICS")
print(f"{'='*60}")
for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
    mean_val = summary_df[summary_df['Metric'] == metric]['Mean'].values[0]
    std_val = summary_df[summary_df['Metric'] == metric]['Std'].values[0]
    print(f"{metric:<12}: {mean_val:.4f} ± {std_val:.4f}")

# =====================
# SAVE RESULTS
# =====================
print(f"\n{'='*60}")
print("SAVING RESULTS")
print(f"{'='*60}")

# Save detailed fold results
cv_df.to_csv("results/cv_detailed_results.csv", index=False)
print("✅ Detailed results saved to results/cv_detailed_results.csv")

# Save summary statistics
summary_df.to_csv("results/cv_summary_stats.csv", index=False)
print("✅ Summary statistics saved to results/cv_summary_stats.csv")

# =====================
# VISUALIZATIONS
# =====================
print("\nGenerating visualizations...")

# 1. Metrics across folds
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Cross Validation Metrics Across Folds - Logistic Regression', 
             fontsize=16, fontweight='bold')

metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
titles = ['Accuracy', 'Precision', 'Recall ⭐', 'F1-Score', 'ROC-AUC']

for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[idx // 3, idx % 3]
    
    values = cv_results[metric]
    mean_val = np.mean(values)
    
    ax.plot(range(1, n_splits + 1), values, marker='o', linewidth=2, 
            markersize=8, color='steelblue', label='Fold Score')
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.4f}')
    ax.set_xlabel('Fold', fontsize=10)
    ax.set_ylabel(title, fontsize=10)
    ax.set_title(f'{title} per Fold', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(1, n_splits + 1))

# Remove empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('results/cv_metrics_per_fold.png', dpi=300, bbox_inches='tight')
print("✅ Metrics plot saved to results/cv_metrics_per_fold.png")
plt.close()

# 2. Box plot of metrics
fig, ax = plt.subplots(figsize=(10, 6))

metrics_data = [
    cv_results['accuracy'],
    cv_results['precision'],
    cv_results['recall'],
    cv_results['f1_score'],
    cv_results['roc_auc']
]

box = ax.boxplot(metrics_data, labels=['Accuracy', 'Precision', 'Recall ⭐', 'F1-Score', 'ROC-AUC'],
                 patch_artist=True, notch=True, showmeans=True)

# Customize box plot
for patch in box['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

for median in box['medians']:
    median.set_color('red')
    median.set_linewidth(2)

for mean in box['means']:
    mean.set_marker('D')
    mean.set_markerfacecolor('green')
    mean.set_markersize(8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Distribution of CV Metrics - Logistic Regression', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('results/cv_metrics_boxplot.png', dpi=300, bbox_inches='tight')
print("✅ Box plot saved to results/cv_metrics_boxplot.png")
plt.close()

# 3. Confusion Matrix Heatmap (Average)
avg_cm = np.array([
    [np.mean(cv_results['tn']), np.mean(cv_results['fp'])],
    [np.mean(cv_results['fn']), np.mean(cv_results['tp'])]
])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            ax=ax)
ax.set_title('Average Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)

plt.tight_layout()
plt.savefig('results/cv_avg_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Average confusion matrix saved to results/cv_avg_confusion_matrix.png")
plt.close()

# 4. Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(titles))
width = 0.15

for i in range(n_splits):
    values = [cv_results[metric][i] for metric in metrics_to_plot]
    ax.bar(x + i * width, values, width, label=f'Fold {i+1}', alpha=0.8)

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Cross Validation Metrics - All Folds Comparison (Logistic Regression)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(titles)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('results/cv_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Metrics comparison saved to results/cv_metrics_comparison.png")
plt.close()

# =====================
# FINAL SUMMARY
# =====================
print(f"\n{'='*60}")
print("CROSS VALIDATION COMPLETED!")
print(f"{'='*60}")

print(f"\n📊 Summary of {n_splits}-Fold Cross Validation:")
print(f"   Model: Logistic Regression (max_iter=1000)")
print(f"   Dataset: CDC BRFSS 2015 (Age 18-44)")
print(f"   Data: {X.shape[0]} samples")
print(f"   Features: {len(SELECTED_FEATURES)} FIXED features (same as deployment)")
print(f"   Preprocessing: StandardScaler + SMOTE per fold")

print(f"\n🎯 Average Performance:")
print(f"   - Accuracy  : {np.mean(cv_results['accuracy']):.4f} ± {np.std(cv_results['accuracy']):.4f}")
print(f"   - Precision : {np.mean(cv_results['precision']):.4f} ± {np.std(cv_results['precision']):.4f}")
print(f"   - Recall    : {np.mean(cv_results['recall']):.4f} ± {np.std(cv_results['recall']):.4f} ⭐")
print(f"   - F1-Score  : {np.mean(cv_results['f1_score']):.4f} ± {np.std(cv_results['f1_score']):.4f}")
print(f"   - ROC-AUC   : {np.mean(cv_results['roc_auc']):.4f} ± {np.std(cv_results['roc_auc']):.4f}")

print(f"\n📌 Fixed Features Used:")
for i, feat in enumerate(SELECTED_FEATURES, 1):
    print(f"   {i:2d}. {feat}")

print(f"\n📁 Results saved to:")
print(f"   - results/cv_detailed_results.csv")
print(f"   - results/cv_summary_stats.csv")
print(f"   - results/cv_metrics_per_fold.png")
print(f"   - results/cv_metrics_boxplot.png")
print(f"   - results/cv_avg_confusion_matrix.png")
print(f"   - results/cv_metrics_comparison.png")

print(f"\n{'='*60}")
print("✅ CV VALIDATION MATCHES DEPLOYMENT MODEL!")
print(f"{'='*60}")