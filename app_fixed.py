import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from fe_pipeline_fixed import preprocess_user_input, validate_input

app = Flask(__name__)

# =====================
# LOAD MODEL & PREPROCESSING OBJECTS
# =====================
print("Loading model and preprocessing objects...")
model = joblib.load("models/diabetes_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_info = joblib.load("models/feature_info.pkl")

selected_features = feature_info['selected_features']
model_type = feature_info.get('model_type', 'LogisticRegression')

print(f"✅ Model loaded successfully!")
print(f"✅ Model type: {model_type}")
print(f"✅ Selected features ({len(selected_features)}): {selected_features}")


@app.route('/')
def index():
    """Landing page dengan disclaimer"""
    return render_template('disclaimer.html')


@app.route("/form")
def form():
    """Form input prediksi diabetes"""
    return render_template("index.html", 
                         result=None, 
                         selected_features=selected_features)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint prediksi diabetes dengan 10 fitur fixed
    """
    
    try:
        # =====================
        # AMBIL INPUT USER (HANYA 10 FITUR)
        # =====================
        input_data = {}
        
        for feature in selected_features:
            value = request.form.get(feature)
            if value is not None and value != '':
                try:
                    input_data[feature] = float(value)
                except ValueError:
                    return render_template("index.html", 
                                         result=None, 
                                         error=f"Format input tidak valid untuk {feature}",
                                         selected_features=selected_features)
            else:
                return render_template("index.html", 
                                     result=None, 
                                     error=f"Field {feature} harus diisi",
                                     selected_features=selected_features)
        
        # =====================
        # VALIDASI INPUT
        # =====================
        is_valid, error_msg = validate_input(input_data, selected_features)
        if not is_valid:
            return render_template("index.html", 
                                 result=None, 
                                 error=error_msg,
                                 selected_features=selected_features)
        
        # =====================
        # PREPROCESSING
        # =====================
        # Bentuk DataFrame dengan urutan fitur yang benar
        user_input = pd.DataFrame([input_data])[selected_features]
        
        # Scale (scaler sudah di-fit pada 10 fitur yang sama)
        X_user_scaled = scaler.transform(user_input)
        
        # =====================
        # PREDIKSI
        # =====================
        prediction = model.predict(X_user_scaled)[0]
        probability = model.predict_proba(X_user_scaled)[0]
        
        # =====================
        # RETURN RESULT
        # =====================
        result_data = {
            'prediction': int(prediction),
            'risk_level': 'high' if prediction == 1 else 'low',
            'probability': {
                'no_diabetes': float(probability[0]),
                'diabetes': float(probability[1])
            }
        }
        
        return render_template("index.html", 
                             result=result_data,
                             selected_features=selected_features)
    
    except Exception as e:
        error_msg = f"Terjadi kesalahan: {str(e)}"
        print(f"Error: {error_msg}")
        import traceback
        traceback.print_exc()
        return render_template("index.html", 
                             result=None, 
                             error=error_msg,
                             selected_features=selected_features)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API endpoint untuk prediksi (JSON format)
    """
    try:
        data = request.get_json()
        
        # Validasi input
        is_valid, error_msg = validate_input(data, selected_features)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Bentuk dataframe dengan urutan yang benar
        user_input = pd.DataFrame([data])[selected_features]
        
        # Preprocessing
        X_user_scaled = scaler.transform(user_input)
        
        # Prediksi
        prediction = model.predict(X_user_scaled)[0]
        probability = model.predict_proba(X_user_scaled)[0]
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'risk_level': 'high' if prediction == 1 else 'low',
            'probability': {
                'no_diabetes': float(probability[0]),
                'diabetes': float(probability[1])
            },
            'features_used': selected_features
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': model_type,
        'dataset': 'CDC BRFSS 2015',
        'features': len(selected_features),
        'selected_features': selected_features
    })


@app.route("/features")
def features():
    """Endpoint untuk melihat feature info"""
    return jsonify({
        'selected_features': selected_features,
        'n_features': len(selected_features),
        'model_type': model_type,
        'coefficients': feature_info.get('coefficients', [])
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
