import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =====================
# FIXED FEATURES (HASIL EKSPERIMEN)
# =====================
SELECTED_FEATURES = [
    'HighBP', 'HighChol', 'BMI', 'AnyHealthcare', 
    'GenHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Income'
]

print("="*60)
print("DIABETES PREDICTION MODEL - LOGISTIC REGRESSION")
print("="*60)
print(f"\n🎯 Fixed Features ({len(SELECTED_FEATURES)}):")
for i, feat in enumerate(SELECTED_FEATURES, 1):
    print(f"  {i}. {feat}")

# =====================
# LOAD DATA
# =====================
print("\n" + "="*60)
print("LOADING DATASET")
print("="*60)
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
print("\n" + "="*60)
print("TARGET DISTRIBUTION")
print("="*60)
print(df['diabetes'].value_counts())
print("\nPercentage:")
print(df['diabetes'].value_counts(normalize=True) * 100)

# =====================
# SELECT ONLY FIXED FEATURES
# =====================
print("\n" + "="*60)
print("SELECTING FIXED FEATURES")
print("="*60)

# Target
y = df['diabetes']

# Pilih hanya 10 fitur yang sudah ditentukan
X = df[SELECTED_FEATURES].copy()

print(f"Features used: {list(X.columns)}")
print(f"Total features: {X.shape[1]}")

# =====================
# TRAIN TEST SPLIT
# =====================
print("\n" + "="*60)
print("SPLITTING DATA")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# =====================
# SCALING
# =====================
print("\n" + "="*60)
print("SCALING FEATURES")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Scaler fitted on {X_train.shape[1]} features")
print(f"Mean after scaling: {X_train_scaled.mean():.6f}")
print(f"Std after scaling: {X_train_scaled.std():.6f}")

# =====================
# HANDLE IMBALANCED DATA WITH SMOTE
# =====================
print("\n" + "="*60)
print("APPLYING SMOTE")
print("="*60)

print(f"Before SMOTE - Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")
print(f"Ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE - Class 0: {(y_train_balanced == 0).sum()}, Class 1: {(y_train_balanced == 1).sum()}")
print(f"Ratio: {(y_train_balanced == 0).sum() / (y_train_balanced == 1).sum():.2f}:1")

# =====================
# TRAIN MODEL (LOGISTIC REGRESSION)
# =====================
print("\n" + "="*60)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("="*60)

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_balanced, y_train_balanced)
print("✅ Model training completed!")

# =====================
# CROSS-VALIDATION
# =====================
print("\n" + "="*60)
print("CROSS-VALIDATION")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=cv, scoring='recall')

print(f"CV Recall: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"CV Scores: {cv_scores}")

# =====================
# EVALUATION
# =====================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Predict
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f} ⭐ (Metric utama)")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)
print(f"True Negative  (TN): {tn:>5}")
print(f"False Positive (FP): {fp:>5}")
print(f"False Negative (FN): {fn:>5}")
print(f"True Positive  (TP): {tp:>5}")
print("="*60)

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_test_pred))

# =====================
# FEATURE COEFFICIENTS
# =====================
print("\n" + "="*60)
print("FEATURE COEFFICIENTS (LOGISTIC REGRESSION)")
print("="*60)

coefficients = pd.DataFrame({
    'Feature': SELECTED_FEATURES,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

print(coefficients.to_string(index=False))

# =====================
# VISUALIZATIONS
# =====================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# 1. ROC Curve
plt.figure(figsize=(10, 6))
fpr, tpr, _ = roc_curve(y_test, y_test_proba)

plt.plot(fpr, tpr, linewidth=2, label=f'Logistic Regression (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Logistic Regression (Test Set)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
print("✅ ROC curve saved to results/roc_curve.png")
plt.close()

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix - Logistic Regression', fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved to results/confusion_matrix.png")
plt.close()

# 3. Feature Coefficients Chart
plt.figure(figsize=(10, 8))
colors = ['green' if x > 0 else 'red' for x in coefficients['Coefficient']]
plt.barh(range(len(coefficients)), coefficients['Coefficient'], color=colors, alpha=0.8)
plt.yticks(range(len(coefficients)), coefficients['Feature'])
plt.xlabel('Coefficient Value', fontsize=12)
plt.title(f'Feature Coefficients - Logistic Regression', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('results/feature_coefficients.png', dpi=300, bbox_inches='tight')
print("✅ Feature coefficients saved to results/feature_coefficients.png")
plt.close()

# =====================
# SAVE MODEL & SCALER
# =====================
print("\n" + "="*60)
print("SAVING MODEL & PREPROCESSING OBJECTS")
print("="*60)

joblib.dump(model, "models/diabetes_model.pkl")
print("✅ Logistic Regression model saved to models/diabetes_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")
print("✅ Scaler saved to models/scaler.pkl")

# Save feature info
feature_info = {
    'selected_features': SELECTED_FEATURES,
    'n_features': len(SELECTED_FEATURES),
    'model_type': 'LogisticRegression',
    'coefficients': coefficients.to_dict('records')
}
joblib.dump(feature_info, "models/feature_info.pkl")
print("✅ Feature info saved to models/feature_info.pkl")

# =====================
# FINAL SUMMARY
# =====================
print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\n📊 Model Performance Summary:")
print(f"   - Model: Logistic Regression")
print(f"   - Dataset: CDC BRFSS 2015 (Age 18-44)")
print(f"   - Total samples: {df.shape[0]}")
print(f"   - Features used: {len(SELECTED_FEATURES)} (FIXED)")
print(f"   - Test Accuracy: {accuracy:.4f}")
print(f"   - Test Precision: {precision:.4f}")
print(f"   - Test Recall: {recall:.4f} ⭐")
print(f"   - Test F1-Score: {f1:.4f}")
print(f"   - Test ROC-AUC: {roc_auc:.4f}")
print(f"   - CV Recall: {cv_scores.mean():.4f}")
print(f"\n🎯 Model ready for deployment!")