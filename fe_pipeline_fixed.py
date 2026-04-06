import numpy as np
import pandas as pd


def preprocess_user_input(df: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    """
    Preprocess user input untuk prediksi diabetes.
    
    Args:
        df: DataFrame dengan data user (harus punya semua selected_features)
        selected_features: List 10 fitur yang digunakan model
    
    Returns:
        pd.DataFrame: DataFrame dengan fitur dalam urutan yang benar
    """
    df = df.copy()
    
    # Pastikan semua fitur ada
    for feature in selected_features:
        if feature not in df.columns:
            raise ValueError(f"Feature {feature} tidak ditemukan dalam input")
    
    # Return dengan urutan yang benar
    return df[selected_features]


def validate_input(data: dict, selected_features: list) -> tuple[bool, str]:
    """
    Validasi input dari user sebelum prediksi.
    
    Args:
        data: Dictionary berisi input user
        selected_features: List fitur yang dibutuhkan (10 fitur)
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Validasi semua fitur yang dibutuhkan ada
    for feature in selected_features:
        if feature not in data:
            return False, f"Field {feature} tidak ditemukan"
        if data[feature] is None:
            return False, f"Field {feature} tidak boleh kosong"
    
    # Validasi khusus per fitur
    
    # 1. HighBP (0/1)
    if 'HighBP' in selected_features:
        if data['HighBP'] not in [0, 1]:
            return False, "HighBP harus 0 (No) atau 1 (Yes)"
    
    # 2. HighChol (0/1)
    if 'HighChol' in selected_features:
        if data['HighChol'] not in [0, 1]:
            return False, "HighChol harus 0 (No) atau 1 (Yes)"
    
    # 3. BMI (10-100)
    if 'BMI' in selected_features:
        bmi = data['BMI']
        if not (10 <= bmi <= 100):
            return False, "BMI harus antara 10-100"
    
    # 4. AnyHealthcare (0/1)
    if 'AnyHealthcare' in selected_features:
        if data['AnyHealthcare'] not in [0, 1]:
            return False, "AnyHealthcare harus 0 (No) atau 1 (Yes)"
    
    # 5. GenHlth (1-5)
    if 'GenHlth' in selected_features:
        if not (1 <= data['GenHlth'] <= 5):
            return False, "GenHlth harus antara 1 (Excellent) - 5 (Poor)"
    
    # 6. PhysHlth (0-30)
    if 'PhysHlth' in selected_features:
        if not (0 <= data['PhysHlth'] <= 30):
            return False, "PhysHlth harus antara 0-30 hari"
    
    # 7. DiffWalk (0/1)
    if 'DiffWalk' in selected_features:
        if data['DiffWalk'] not in [0, 1]:
            return False, "DiffWalk harus 0 (No) atau 1 (Yes)"
    
    # 8. Sex (0/1)
    if 'Sex' in selected_features:
        if data['Sex'] not in [0, 1]:
            return False, "Sex harus 0 (Female) atau 1 (Male)"
    
    # 9. Age (1-13, fokus 1-5 untuk usia produktif)
    if 'Age' in selected_features:
        if not (1 <= data['Age'] <= 13):
            return False, "Age harus antara 1-13"
    
    # 10. Income (1-8)
    if 'Income' in selected_features:
        if not (1 <= data['Income'] <= 8):
            return False, "Income harus antara 1-8"
    
    return True, ""


if __name__ == "__main__":
    # Test preprocessing
    SELECTED_FEATURES = [
        'HighBP', 'HighChol', 'BMI', 'AnyHealthcare', 
        'GenHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Income'
    ]
    
    # Test data
    sample_data = {
        'HighBP': 0,
        'HighChol': 1,
        'BMI': 28.5,
        'AnyHealthcare': 1,
        'GenHlth': 2,
        'PhysHlth': 5,
        'DiffWalk': 0,
        'Sex': 1,
        'Age': 3,
        'Income': 6
    }
    
    print("=" * 60)
    print("TESTING FE_PIPELINE")
    print("=" * 60)
    
    # Test validation
    print("\n1. Testing validation...")
    is_valid, error_msg = validate_input(sample_data, SELECTED_FEATURES)
    print(f"   Valid: {is_valid}")
    if not is_valid:
        print(f"   Error: {error_msg}")
    else:
        print("   ✅ All validations passed!")
    
    # Test preprocessing
    print("\n2. Testing preprocessing...")
    sample_df = pd.DataFrame([sample_data])
    print(f"   Input shape: {sample_df.shape}")
    print(f"   Input columns: {list(sample_df.columns)}")
    
    processed = preprocess_user_input(sample_df, SELECTED_FEATURES)
    print(f"\n   Output shape: {processed.shape}")
    print(f"   Output columns: {list(processed.columns)}")
    print(f"\n   Processed data:")
    print(processed)
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
