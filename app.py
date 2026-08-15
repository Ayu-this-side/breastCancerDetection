import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Base Directory & Explicit Template / Static Paths for VS Code execution safety
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# Standard Breast Cancer Dataset 10 Mean Features
FEATURE_NAMES = [
    'mean_radius',
    'mean_texture',
    'mean_perimeter',
    'mean_area',
    'mean_smoothness',
    'mean_compactness',
    'mean_concavity',
    'mean_concave_points',
    'mean_symmetry',
    'mean_fractal_dimension'
]

FEATURE_LABELS = {
    'mean_radius': 'Mean Radius (mm)',
    'mean_texture': 'Mean Texture (Standard Deviation)',
    'mean_perimeter': 'Mean Perimeter (mm)',
    'mean_area': 'Mean Area (mm²)',
    'mean_smoothness': 'Mean Smoothness (Local Variation)',
    'mean_compactness': 'Mean Compactness',
    'mean_concavity': 'Mean Concavity',
    'mean_concave_points': 'Mean Concave Points',
    'mean_symmetry': 'Mean Symmetry',
    'mean_fractal_dimension': 'Mean Fractal Dimension'
}

# Global references for loaded model and scaler
model = None
scaler = None
model_metrics = {}
population_stats = {}

def train_default_baseline_model():
    """
    Trains a high-performance baseline Random Forest classifier on Scikit-Learn's
    Breast Cancer Wisconsin Dataset if the user hasn't uploaded a custom model.pkl yet.
    """
    global model, scaler, model_metrics, population_stats

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Select top 10 mean features for intuitive web UI interaction
    df_10 = df[[f for f in data.feature_names if 'mean' in f or f in FEATURE_NAMES]].copy()
    df_10.columns = [col.replace(' ', '_') for col in df_10.columns]
    
    # 0 = Malignant in sklearn default, 1 = Benign
    # Let's map target so 1 = Malignant (Risk detected) and 0 = Benign (Normal)
    y = np.where(data.target == 0, 1, 0)
    X = df_10.iloc[:, :10]

    # Calculate Population Baselines (Benign vs Malignant means)
    benign_means = X[y == 0].mean().to_dict()
    malignant_means = X[y == 1].mean().to_dict()
    overall_min = X.min().to_dict()
    overall_max = X.max().to_dict()

    population_stats = {
        'benign_means': benign_means,
        'malignant_means': malignant_means,
        'min_values': overall_min,
        'max_values': overall_max
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler_obj = StandardScaler()
    X_train_scaled = scaler_obj.fit_transform(X_train)
    X_test_scaled = scaler_obj.transform(X_test)

    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))

    # Save to disk
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler_obj, SCALER_PATH)

    model = clf
    scaler = scaler_obj

    importances = clf.feature_importances_.tolist()
    model_metrics = {
        'accuracy': round(acc * 100, 2),
        'roc_auc': round(auc * 100, 2),
        'model_type': 'Random Forest Classifier (Baseline Dataset)',
        'feature_importances': dict(zip(X.columns, [round(val, 4) for val in importances]))
    }
    print(f"[*] Baseline Model trained & saved. Accuracy: {acc*100:.2f}%, ROC-AUC: {auc*100:.2f}%")

def load_or_init_model():
    """Loads saved model & scaler from disk or trains baseline if not found."""
    global model, scaler, model_metrics, population_stats
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("[+] Successfully loaded custom/saved model and scaler from model/ directory.")
            # Populate basic metrics & stats
            train_default_baseline_model()
        except Exception as e:
            print(f"[!] Error loading model file: {e}. Falling back to baseline model training...")
            train_default_baseline_model()
    else:
        print("[*] No custom model found in model/ directory. Training baseline model...")
        train_default_baseline_model()

# Initialize model on app startup
load_or_init_model()

@app.route('/')
def index():
    """Renders the primary web interface."""
    return render_template('index.html')

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Returns model metadata, accuracy, and baseline benchmarks."""
    return jsonify({
        'status': 'success',
        'metrics': model_metrics,
        'features': FEATURE_LABELS,
        'population_stats': population_stats
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accepts feature input JSON payload and computes risk probability & category.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400

        # Extract 10 required feature values
        input_values = []
        feature_dict = {}
        for feat in FEATURE_NAMES:
            val = float(data.get(feat, 0.0))
            input_values.append(val)
            feature_dict[feat] = val

        input_arr = np.array([input_values])
        
        # Apply scaling if scaler exists
        if scaler is not None:
            input_scaled = scaler.transform(input_arr)
        else:
            input_scaled = input_arr

        # Predict probability
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[0]
            # Assuming index 1 corresponds to Malignant risk
            malignant_prob = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        else:
            prediction_raw = int(model.predict(input_scaled)[0])
            malignant_prob = 0.95 if prediction_raw == 1 else 0.05

        benign_prob = 1.0 - malignant_prob

        # Category Determination
        if malignant_prob >= 0.70:
            risk_level = 'High Risk'
            diagnosis = 'Malignant (Positive)'
            risk_color = '#FF2E93'  # Glowing magenta-red
            confidence_tag = 'Strong Signal'
        elif malignant_prob >= 0.35:
            risk_level = 'Moderate Risk'
            diagnosis = 'Inconclusive / Borderline'
            risk_color = '#FFB800'  # Glowing amber
            confidence_tag = 'Requires Further Clinical Review'
        else:
            risk_level = 'Low Risk'
            diagnosis = 'Benign (Negative)'
            risk_color = '#00E699'  # Emerald neon
            confidence_tag = 'High Confidence Benign'

        # Identify anomalous features compared to benign baselines
        anomalies = []
        benign_means = population_stats.get('benign_means', {})
        malignant_means = population_stats.get('malignant_means', {})

        for feat in FEATURE_NAMES:
            val = feature_dict[feat]
            b_mean = benign_means.get(feat, val)
            m_mean = malignant_means.get(feat, val)
            
            # Check deviation
            if b_mean > 0:
                dev_pct = ((val - b_mean) / b_mean) * 100
                if dev_pct > 25:
                    anomalies.append({
                        'feature': feat,
                        'label': FEATURE_LABELS.get(feat, feat),
                        'user_val': val,
                        'benign_avg': round(b_mean, 4),
                        'malignant_avg': round(m_mean, 4),
                        'elevation_pct': round(dev_pct, 1)
                    })

        return jsonify({
            'status': 'success',
            'prediction': {
                'diagnosis': diagnosis,
                'is_malignant': malignant_prob >= 0.5,
                'malignant_probability': round(malignant_prob * 100, 2),
                'benign_probability': round(benign_prob * 100, 2),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'confidence_tag': confidence_tag
            },
            'anomalies': sorted(anomalies, key=lambda x: x['elevation_pct'], reverse=True)[:4],
            'input_features': feature_dict
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  ONCOVISION - BREAST CANCER DIAGNOSTIC & PREDICTION WEB SERVER")
    print("="*60)
    print("  Running on: http://127.0.0.1:5000")
    print(f"  Project Base Directory: {BASE_DIR}")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
