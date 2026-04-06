import pandas as pd
from sklearn.model_selection import train_test_split

# FIXED FEATURES (SAMA SEPERTI train.py)
SELECTED_FEATURES = [
    'HighBP', 'HighChol', 'BMI', 'AnyHealthcare', 
    'GenHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Income'
]

print("="*60)
print("EXPORT TEST SET TO CSV")
print("="*60)

# LOAD DATA (SAMA SEPERTI train.py)
print("\nLoading dataset...")
df = pd.read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv")
df = df.rename(columns={'Diabetes_binary': 'diabetes'})
print(f"Dataset loaded: {df.shape}")

# FILTER POPULATION (USIA PRODUKTIF 18-44 tahun)
print("\nFiltering age 18-44 (fokus usia produktif)...")
print(f"Before filter: {df.shape}")
df = df[df['Age'] <= 5].copy()  # Age 1-5 = 18-44 tahun
print(f"After filter: {df.shape}")

# DROP DUPLICATES
print(f"\nDuplicates before: {df.duplicated().sum()}")
df = df.drop_duplicates()
print(f"After drop duplicates: {df.shape}")

# SELECT FIXED FEATURES
print("\n" + "="*60)
print("SELECTING FEATURES")
print("="*60)

y = df['diabetes']
X = df[SELECTED_FEATURES].copy()

print(f"Features selected: {list(X.columns)}")
print(f"Total features: {X.shape[1]}")

# TRAIN TEST SPLIT (SAMA SEPERTI train.py)
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

# PREPARE TEST SET FOR EXPORT
print("\n" + "="*60)
print("PREPARING TEST SET")
print("="*60)

# Gabungkan X_test dan y_test
test_set = X_test.copy()
test_set['diabetes'] = y_test

print(f"Test set shape: {test_set.shape}")
print(f"Columns: {list(test_set.columns)}")

# Check distribution
print("\nTarget distribution in test set:")
print(test_set['diabetes'].value_counts())
print("\nPercentage:")
print(test_set['diabetes'].value_counts(normalize=True) * 100)

# EXPORT TO CSV
print("\n" + "="*60)
print("EXPORTING TO CSV")
print("="*60)

output_file = "data/test_set.csv"
test_set.to_csv(output_file, index=False)

print(f"✅ Test set exported to: {output_file}")
print(f"   - Total samples: {len(test_set)}")
print(f"   - Features: {len(SELECTED_FEATURES)}")
print(f"   - Target column: diabetes")

# CREATE SUMMARY FILE
summary = {
    'total_samples': len(test_set),
    'n_features': len(SELECTED_FEATURES),
    'features': SELECTED_FEATURES,
    'target_column': 'diabetes',
    'class_0_count': int((test_set['diabetes'] == 0).sum()),
    'class_1_count': int((test_set['diabetes'] == 1).sum()),
    'test_size_ratio': 0.2,
    'random_state': 42,
    'stratified': True
}

import json

summary_file = "data/test_set_info.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Test set info saved to: {summary_file}")

# DISPLAY SAMPLE
print("\n" + "="*60)
print("SAMPLE DATA (First 10 rows)")
print("="*60)
print(test_set.head(10).to_string(index=False))

print("\n" + "="*60)
print("EXPORT COMPLETED!")
print("="*60)
print(f"\n📁 Files created:")
print(f"   1. {output_file}")
print(f"   2. {summary_file}")
print(f"\n✅ Test set is ready for evaluation!")
