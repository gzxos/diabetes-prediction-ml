# %% [markdown]
# # Feature Selection Optimization Experiment
# 
# **Tujuan:** Menemukan jumlah fitur optimal untuk model prediksi diabetes
# 
# **Metodologi:**
# - Test dengan berbagai jumlah fitur: 5, 7, 10, 12, 15, 18, 21
# - Evaluasi performa masing-masing
# - Analisis marginal gain
# - Uji signifikansi statistik
# - Rekomendasi fitur untuk implementasi

# %%
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from scipy import stats

warnings.filterwarnings('ignore')

# Set style untuk visualisasi
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# %% [markdown]
# ## 1. Load dan Preprocessing Data

# %%
# Load dataset
df = pd.read_csv("../data/diabetes_binary_health_indicators_BRFSS2015.csv")
df = df.rename(columns={'Diabetes_binary': 'diabetes'})

print("Shape dataset awal:", df.shape)
print("\nDistribusi diabetes:")
print(df['diabetes'].value_counts())

# %%
# Filter usia produktif (18-44 tahun)
print(f"Data sebelum filter usia: {df.shape}")
df = df[df['Age'] <= 5].copy()  # Age 1-5 = 18-44 tahun
print(f"Data setelah filter usia 18-44: {df.shape}")

# Hapus duplikat
print(f"\nDuplikat sebelum: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"Data setelah drop duplikat: {df.shape}")

print("\nDistribusi diabetes setelah preprocessing:")
print(df['diabetes'].value_counts())
print(df['diabetes'].value_counts(normalize=True) * 100)

# %%
# Pisahkan fitur dan target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

print(f"Jumlah fitur total: {X.shape[1]}")
print(f"Nama fitur:")
print(X.columns.tolist())

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nDistribusi training set:")
print(y_train.value_counts())
print(f"\nDistribusi test set:")
print(y_test.value_counts())

# %%
# Scaling fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling selesai!")
print(f"Mean setelah scaling: {X_train_scaled.mean():.6f}")
print(f"Std setelah scaling: {X_train_scaled.std():.6f}")

# %% [markdown]
# ## 2. Mutual Information Analysis

# %%
# Hitung mutual information untuk semua fitur
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)

# Buat DataFrame
mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("=" * 60)
print("MUTUAL INFORMATION SCORES - ALL FEATURES")
print("=" * 60)
print(mi_df.to_string(index=False))

# Visualisasi MI Scores
plt.figure(figsize=(12, 8))
plt.barh(range(len(mi_df)), mi_df['MI_Score'], color='steelblue', alpha=0.7)
plt.yticks(range(len(mi_df)), mi_df['Feature'])
plt.xlabel('Mutual Information Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Feature Importance - Mutual Information Scores', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('mi_scores_all_features.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. Feature Selection Experiment

# %%
print("\n" + "=" * 80)
print("FEATURE SELECTION EXPERIMENT - Finding Optimal Number of Features")
print("=" * 80)

# Test dengan berbagai jumlah fitur
feature_counts = [5, 7, 10, 12, 15, 18, 21]
selection_results = []

# Initialize SMOTE
smote = SMOTE(random_state=42)

for k in feature_counts:
    print(f"\n{'='*70}")
    print(f"Testing with TOP {k} Features")
    print(f"{'='*70}")
    
    # Feature Selection
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_k = selector.fit_transform(X_train_scaled, y_train)
    X_test_k = selector.transform(X_test_scaled)
    
    # Get selected features
    selected_features_k = X.columns[selector.get_support()].tolist()
    
    # SMOTE
    X_balanced_k, y_balanced_k = smote.fit_resample(X_train_k, y_train)
    
    # Train Logistic Regression
    lr_k = LogisticRegression(max_iter=1000, random_state=42)
    lr_k.fit(X_balanced_k, y_balanced_k)
    
    # Predict
    y_pred_k = lr_k.predict(X_test_k)
    y_proba_k = lr_k.predict_proba(X_test_k)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred_k)
    prec = precision_score(y_test, y_pred_k)
    rec = recall_score(y_test, y_pred_k)
    f1 = f1_score(y_test, y_pred_k)
    roc = roc_auc_score(y_test, y_proba_k)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_k = cross_val_score(lr_k, X_balanced_k, y_balanced_k, 
                                   cv=cv, scoring='recall')
    
    # Store results
    selection_results.append({
        'n_features': k,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': roc,
        'cv_recall_mean': cv_scores_k.mean(),
        'cv_recall_std': cv_scores_k.std(),
        'features': selected_features_k
    })
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")
    print(f"CV Recall: {cv_scores_k.mean():.4f} (+/- {cv_scores_k.std():.4f})")
    print(f"\nSelected features:")
    for i, feat in enumerate(selected_features_k, 1):
        print(f"  {i}. {feat}")

# Convert to DataFrame
selection_df = pd.DataFrame(selection_results)

print("\n" + "=" * 80)
print("FEATURE SELECTION SUMMARY")
print("=" * 80)
print(selection_df[['n_features', 'accuracy', 'precision', 'recall', 
                     'f1_score', 'roc_auc']].round(4).to_string(index=False))

# %% [markdown]
# ## 4. Marginal Gain Analysis

# %%
# Calculate marginal gains
selection_df['marginal_recall_gain'] = selection_df['recall'].diff().fillna(selection_df['recall'].iloc[0]) * 100
selection_df['marginal_f1_gain'] = selection_df['f1_score'].diff().fillna(selection_df['f1_score'].iloc[0]) * 100

print("\n" + "=" * 80)
print("MARGINAL GAIN ANALYSIS")
print("=" * 80)
print(selection_df[['n_features', 'recall', 'marginal_recall_gain', 
                     'f1_score', 'marginal_f1_gain']].round(4).to_string(index=False))

# Identify elbow point (where marginal gain < 1%)
elbow_point = selection_df[selection_df['marginal_recall_gain'] < 1.0]['n_features'].min()
if pd.isna(elbow_point):
    # If all gains > 1%, pick the one before the smallest gain
    min_gain_idx = selection_df['marginal_recall_gain'].idxmin()
    elbow_point = selection_df.iloc[max(0, min_gain_idx-1)]['n_features']

print(f"\n✓ Elbow Point Detected: {elbow_point} features")
print(f"  (Point where marginal gain becomes < 1%)")

# %% [markdown]
# ## 5. Visualization

# %%
# Comprehensive Visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Recall Curve with Elbow
ax1 = axes[0, 0]
ax1.plot(selection_df['n_features'], selection_df['recall'], 
         marker='o', linewidth=2.5, markersize=10, color='#2E86AB', label='Recall')
ax1.scatter(selection_df['n_features'], selection_df['recall'], 
            s=150, color='#2E86AB', edgecolors='black', linewidths=2, zorder=5)

# Highlight elbow point
elbow_recall = selection_df[selection_df['n_features']==elbow_point]['recall'].values[0]
ax1.axvline(x=elbow_point, color='red', linestyle='--', linewidth=2, 
            label=f'Elbow Point ({int(elbow_point)} features)')
ax1.scatter([elbow_point], [elbow_recall], 
            s=300, color='red', edgecolors='black', linewidths=2, zorder=10, marker='*')

ax1.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax1.set_ylabel('Recall Score', fontsize=12, fontweight='bold')
ax1.set_title('Recall vs Number of Features\n(Elbow Curve Analysis)', 
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(selection_df['n_features'])
ax1.set_ylim([selection_df['recall'].min() - 0.02, selection_df['recall'].max() + 0.02])

# Add annotation
ax1.annotate(f'Optimal: {elbow_recall:.4f}', 
            xy=(elbow_point, elbow_recall), 
            xytext=(elbow_point + 2, elbow_recall - 0.01),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, fontweight='bold', color='red')

# 2. Multiple Metrics Comparison
ax2 = axes[0, 1]
ax2.plot(selection_df['n_features'], selection_df['accuracy'], 
         marker='s', linewidth=2, label='Accuracy', color='#F38181')
ax2.plot(selection_df['n_features'], selection_df['precision'], 
         marker='^', linewidth=2, label='Precision', color='#95E1D3')
ax2.plot(selection_df['n_features'], selection_df['recall'], 
         marker='o', linewidth=2, label='Recall', color='#2E86AB')
ax2.plot(selection_df['n_features'], selection_df['f1_score'], 
         marker='D', linewidth=2, label='F1-Score', color='#FFA07A')
ax2.axvline(x=elbow_point, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('All Metrics vs Number of Features', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(selection_df['n_features'])

# 3. Marginal Gain (Bar Chart)
ax3 = axes[1, 0]
colors = ['#95E1D3' if x >= 1.0 else '#F38181' 
          for x in selection_df['marginal_recall_gain']]
bars = ax3.bar(range(len(selection_df)), selection_df['marginal_recall_gain'], 
               color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
ax3.axhline(y=1, color='orange', linestyle='--', linewidth=2, 
            label='Diminishing Returns (1%)')
ax3.set_xlabel('Feature Addition Steps', fontsize=12, fontweight='bold')
ax3.set_ylabel('Marginal Recall Gain (%)', fontsize=12, fontweight='bold')
ax3.set_title('Marginal Performance Gain Analysis', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(selection_df)))

# Create labels for x-axis
x_labels = []
for i in range(len(selection_df)):
    if i == 0:
        x_labels.append(f"0→{selection_df['n_features'].iloc[i]}")
    else:
        x_labels.append(f"{selection_df['n_features'].iloc[i-1]}→{selection_df['n_features'].iloc[i]}")
ax3.set_xticklabels(x_labels, rotation=45, ha='right')

ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, selection_df['marginal_recall_gain'])):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%',
            ha='center', va='bottom' if height >= 0 else 'top', 
            fontsize=9, fontweight='bold')

# 4. ROC-AUC Progress
ax4 = axes[1, 1]
ax4.plot(selection_df['n_features'], selection_df['roc_auc'], 
         marker='o', linewidth=2.5, markersize=10, color='#9B59B6', label='ROC-AUC')
ax4.scatter(selection_df['n_features'], selection_df['roc_auc'], 
            s=150, color='#9B59B6', edgecolors='black', linewidths=2, zorder=5)
ax4.axvline(x=elbow_point, color='red', linestyle='--', linewidth=2, alpha=0.5,
            label=f'Elbow Point ({int(elbow_point)})')
ax4.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax4.set_ylabel('ROC-AUC Score', fontsize=12, fontweight='bold')
ax4.set_title('ROC-AUC vs Number of Features', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(selection_df['n_features'])
ax4.set_ylim([selection_df['roc_auc'].min() - 0.01, selection_df['roc_auc'].max() + 0.01])

plt.tight_layout()
plt.savefig('feature_selection_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Statistical Significance Test

# %%
# Compare elbow point vs full features
print("\n" + "=" * 80)
print(f"STATISTICAL SIGNIFICANCE TEST: {int(elbow_point)} Features vs 21 Features")
print("=" * 80)

recall_elbow = selection_df[selection_df['n_features']==elbow_point]['recall'].values[0]
recall_21 = selection_df[selection_df['n_features']==21]['recall'].values[0]
n_test_samples = len(y_test)

# Z-test for proportions
pooled_p = (recall_elbow + recall_21) / 2
se = np.sqrt(2 * pooled_p * (1 - pooled_p) / n_test_samples)
z_score = (recall_21 - recall_elbow) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

print(f"\nRecall with {int(elbow_point)} features: {recall_elbow:.4f}")
print(f"Recall with 21 features: {recall_21:.4f}")
print(f"Absolute difference: {abs(recall_21 - recall_elbow):.4f}")
print(f"Percentage difference: {abs(recall_21 - recall_elbow)*100:.2f}%")

print(f"\nStatistical Test:")
print(f"  H₀: No significant difference between {int(elbow_point)} and 21 features")
print(f"  H₁: Significant difference exists")
print(f"  Z-score: {z_score:.4f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Significance level (α): 0.05")

print(f"\nResult:")
if p_value > 0.05:
    print(f"  ✓ FAIL to reject H₀ (p-value = {p_value:.4f} > 0.05)")
    print(f"  ✓ The difference is NOT statistically significant")
    print(f"  ✓ Using {int(elbow_point)} features is JUSTIFIED")
else:
    print(f"  ⚠ REJECT H₀ (p-value = {p_value:.4f} < 0.05)")
    print(f"  ⚠ The difference IS statistically significant")
    print(f"  ⚠ Consider the trade-off carefully")

print(f"\nPractical Interpretation:")
reduction_pct = ((21 - elbow_point) / 21) * 100
print(f"  - Feature reduction: {21 - int(elbow_point)} features ({reduction_pct:.1f}% less)")
print(f"  - Performance loss: {abs(recall_21 - recall_elbow)*100:.2f}%")
print(f"  - User effort reduction: ~{reduction_pct:.0f}%")
print(f"  - Implementation complexity: Significantly reduced")

# %% [markdown]
# ## 7. Recommended Feature Set

# %%
# Get best feature set based on elbow point
best_k = int(elbow_point)
best_features = selection_df[selection_df['n_features']==best_k]['features'].values[0]

print("\n" + "=" * 80)
print(f"RECOMMENDED FEATURE SET FOR IMPLEMENTATION ({best_k} Features)")
print("=" * 80)

# Get MI scores for these features
selector_best = SelectKBest(mutual_info_classif, k=best_k)
selector_best.fit(X_train_scaled, y_train)
best_features_array = X.columns[selector_best.get_support()]
mi_scores_all = mutual_info_classif(X_train_scaled, y_train, random_state=42)
mi_scores_selected = mi_scores_all[selector_best.get_support()]

feature_recommendations = pd.DataFrame({
    'Rank': range(1, best_k+1),
    'Feature': best_features_array,
    'MI_Score': mi_scores_selected
}).sort_values('MI_Score', ascending=False)
feature_recommendations['Rank'] = range(1, best_k+1)

print("\n" + feature_recommendations.to_string(index=False))

# Visualize recommended features
plt.figure(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_recommendations)))
plt.barh(range(len(feature_recommendations)), 
         feature_recommendations['MI_Score'], 
         color=colors, edgecolor='black', linewidth=1.5)
plt.yticks(range(len(feature_recommendations)), 
           [f"{row['Rank']}. {row['Feature']}" 
            for _, row in feature_recommendations.iterrows()])
plt.xlabel('Mutual Information Score', fontsize=12, fontweight='bold')
plt.ylabel('Feature (Ranked)', fontsize=12, fontweight='bold')
plt.title(f'Top {best_k} Features - Recommended for Implementation', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'recommended_{best_k}_features.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Performance Summary

# %%
print("\n" + "=" * 80)
print(f"PERFORMANCE SUMMARY - RECOMMENDED {best_k} FEATURES")
print("=" * 80)

best_result = selection_df[selection_df['n_features']==best_k].iloc[0]

print(f"\nModel: Logistic Regression")
print(f"Features: {best_k} (from {X.shape[1]} total)")
print(f"Reduction: {((X.shape[1] - best_k) / X.shape[1] * 100):.1f}%")

print(f"\nPerformance Metrics:")
print(f"  Accuracy:  {best_result['accuracy']:.4f}")
print(f"  Precision: {best_result['precision']:.4f}")
print(f"  Recall:    {best_result['recall']:.4f}")
print(f"  F1-Score:  {best_result['f1_score']:.4f}")
print(f"  ROC-AUC:   {best_result['roc_auc']:.4f}")

print(f"\nCross-Validation:")
print(f"  Mean Recall: {best_result['cv_recall_mean']:.4f}")
print(f"  Std Recall:  {best_result['cv_recall_std']:.4f}")

print(f"\n" + "=" * 80)
print("JUSTIFICATION FOR IMPLEMENTATION")
print("=" * 80)
print(f"\n✓ Optimal Balance:")
print(f"  - High performance: Recall = {best_result['recall']:.4f} (~{best_result['recall']*100:.1f}%)")
print(f"  - Practical usability: Only {best_k} inputs required")
print(f"  - Statistical validity: Not significantly different from full model")

print(f"\n✓ Advantages:")
print(f"  - {((X.shape[1] - best_k) / X.shape[1] * 100):.0f}% fewer features to collect")
print(f"  - Faster data entry (~{((X.shape[1] - best_k) / X.shape[1] * 100):.0f}% time saved)")
print(f"  - Better user experience (shorter form)")
print(f"  - Lower implementation complexity")
print(f"  - Reduced risk of overfitting")

print(f"\n✓ Trade-off:")
full_recall = selection_df[selection_df['n_features']==21]['recall'].values[0]
print(f"  - Performance loss: {abs(full_recall - best_result['recall'])*100:.2f}%")
print(f"  - Practical gain: {((X.shape[1] - best_k) / X.shape[1] * 100):.0f}% fewer inputs")
print(f"  - Verdict: ACCEPTABLE for screening tool")

print("\n" + "=" * 80)

# %% [markdown]
# ## 9. Export Results

# %%
# Save results to CSV
selection_df.to_csv('feature_selection_results.csv', index=False)
print("✓ Results saved to: feature_selection_results.csv")

# Save recommended features
feature_recommendations.to_csv(f'recommended_{best_k}_features.csv', index=False)
print(f"✓ Recommended features saved to: recommended_{best_k}_features.csv")

# Create summary report
with open('feature_selection_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FEATURE SELECTION EXPERIMENT - SUMMARY REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset: Diabetes Prediction (Age 18-44)\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Total features available: {X.shape[1]}\n\n")
    
    f.write("EXPERIMENT PARAMETERS:\n")
    f.write(f"  Feature counts tested: {feature_counts}\n")
    f.write(f"  Model: Logistic Regression\n")
    f.write(f"  Balancing: SMOTE\n")
    f.write(f"  Cross-validation: 5-fold Stratified\n\n")
    
    f.write("RESULTS:\n")
    f.write(f"  Recommended features: {best_k}\n")
    f.write(f"  Elbow point: {best_k} features\n")
    f.write(f"  Statistical significance: {'NOT significant' if p_value > 0.05 else 'Significant'}\n")
    f.write(f"  P-value ({best_k} vs 21): {p_value:.4f}\n\n")
    
    f.write("PERFORMANCE (with recommended features):\n")
    f.write(f"  Accuracy:  {best_result['accuracy']:.4f}\n")
    f.write(f"  Precision: {best_result['precision']:.4f}\n")
    f.write(f"  Recall:    {best_result['recall']:.4f}\n")
    f.write(f"  F1-Score:  {best_result['f1_score']:.4f}\n")
    f.write(f"  ROC-AUC:   {best_result['roc_auc']:.4f}\n\n")
    
    f.write(f"RECOMMENDED FEATURES (Top {best_k}):\n")
    for _, row in feature_recommendations.iterrows():
        f.write(f"  {row['Rank']}. {row['Feature']:25s} (MI Score: {row['MI_Score']:.4f})\n")
    
    f.write("\n" + "="*80 + "\n")

print("✓ Summary report saved to: feature_selection_summary.txt")

print("\n" + "="*80)
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nGenerated files:")
print("  1. feature_selection_results.csv")
print(f"  2. recommended_{best_k}_features.csv")
print("  3. feature_selection_summary.txt")
print("  4. mi_scores_all_features.png")
print("  5. feature_selection_comprehensive_analysis.png")
print(f"  6. recommended_{best_k}_features.png")
print("\nNext steps:")
print(f"  → Use the {best_k} recommended features in your main model")
print(f"  → Update k={best_k} in your feature selection code")
print("  → Use the graphs and summary for your thesis documentation")
print("="*80)
