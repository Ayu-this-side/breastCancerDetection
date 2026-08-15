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


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


# Create model directory if it does not exist
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_NAMES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension"
]


FEATURE_LABELS = {
    "mean_radius": "Mean Radius (mm)",
    "mean_texture": "Mean Texture (Standard Deviation)",
    "mean_perimeter": "Mean Perimeter (mm)",
    "mean_area": "Mean Area (mm²)",
    "mean_smoothness": "Mean Smoothness (Local Variation)",
    "mean_compactness": "Mean Compactness",
    "mean_concavity": "Mean Concavity",
    "mean_concave_points": "Mean Concave Points",
    "mean_symmetry": "Mean Symmetry",
    "mean_fractal_dimension": "Mean Fractal Dimension"
}


# ============================================================
# GLOBAL MODEL REFERENCES
# ============================================================

model = None
scaler = None

model_metrics = {}
population_stats = {}


# ============================================================
# DATASET / POPULATION STATISTICS
# ============================================================

def calculate_population_stats():
    """
    Calculate population statistics from the standard
    Scikit-Learn Breast Cancer Wisconsin dataset.

    These statistics are used by the frontend to compare
    user input against benign and malignant averages.
    """

    global population_stats

    data = load_breast_cancer()

    df = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    # Select the 10 "mean" features
    mean_columns = [
        column
        for column in data.feature_names
        if column.startswith("mean ")
    ]

    df_10 = df[mean_columns].copy()

    # Convert:
    # mean radius -> mean_radius
    # mean texture -> mean_texture
    df_10.columns = [
        column.replace(" ", "_")
        for column in df_10.columns
    ]

    # sklearn:
    # 0 = malignant
    # 1 = benign
    #
    # We map:
    # 1 = malignant
    # 0 = benign
    y = np.where(data.target == 0, 1, 0)

    population_stats = {
        "benign_means": (
            df_10[y == 0]
            .mean()
            .to_dict()
        ),

        "malignant_means": (
            df_10[y == 1]
            .mean()
            .to_dict()
        ),

        "min_values": (
            df_10
            .min()
            .to_dict()
        ),

        "max_values": (
            df_10
            .max()
            .to_dict()
        )
    }


# ============================================================
# BASELINE MODEL TRAINING
# ============================================================

def train_default_baseline_model():
    """
    Train a baseline Random Forest model using the
    Scikit-Learn Breast Cancer Wisconsin dataset.

    This function is ONLY used when saved model files
    are not available or cannot be loaded.
    """

    global model
    global scaler
    global model_metrics

    print("[*] Training baseline Random Forest model...")

    data = load_breast_cancer()

    df = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    # Select only the 10 mean features
    mean_columns = [
        column
        for column in data.feature_names
        if column.startswith("mean ")
    ]

    df_10 = df[mean_columns].copy()

    df_10.columns = [
        column.replace(" ", "_")
        for column in df_10.columns
    ]

    X = df_10[FEATURE_NAMES]

    # sklearn:
    # 0 = malignant
    # 1 = benign
    #
    # Our application:
    # 1 = malignant
    # 0 = benign
    y = np.where(data.target == 0, 1, 0)

    # Calculate population statistics
    calculate_population_stats()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Standardization
    scaler_obj = StandardScaler()

    X_train_scaled = scaler_obj.fit_transform(X_train)
    X_test_scaled = scaler_obj.transform(X_test)

    # Random Forest
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        random_state=42
    )

    clf.fit(
        X_train_scaled,
        y_train
    )

    # Predictions
    y_pred = clf.predict(X_test_scaled)

    y_prob = clf.predict_proba(
        X_test_scaled
    )[:, 1]

    # Metrics
    acc = float(
        accuracy_score(
            y_test,
            y_pred
        )
    )

    auc = float(
        roc_auc_score(
            y_test,
            y_prob
        )
    )

    # Save model/scaler
    joblib.dump(
        clf,
        MODEL_PATH
    )

    joblib.dump(
        scaler_obj,
        SCALER_PATH
    )

    # Set global references
    model = clf
    scaler = scaler_obj

    # Feature importance
    importances = clf.feature_importances_.tolist()

    model_metrics = {
        "accuracy": round(acc * 100, 2),
        "roc_auc": round(auc * 100, 2),
        "model_type": "Random Forest Classifier (Baseline Dataset)",
        "feature_importances": dict(
            zip(
                FEATURE_NAMES,
                [
                    round(value, 4)
                    for value in importances
                ]
            )
        )
    }

    print(
        f"[*] Baseline model trained successfully. "
        f"Accuracy: {acc * 100:.2f}% | "
        f"ROC-AUC: {auc * 100:.2f}%"
    )


# ============================================================
# LOAD SAVED MODEL
# ============================================================

def load_or_init_model():
    """
    Load the saved model and scaler if they exist.

    IMPORTANT:
    We do NOT retrain the model when model.pkl exists.

    This prevents Render from replacing the saved model
    every time the server restarts.

    If the model files are missing or invalid, a baseline
    model is trained instead.
    """

    global model
    global scaler
    global model_metrics

    model_exists = os.path.exists(MODEL_PATH)
    scaler_exists = os.path.exists(SCALER_PATH)

    if model_exists and scaler_exists:

        print("[*] Saved model files found.")
        print(f"[*] Loading model from: {MODEL_PATH}")
        print(f"[*] Loading scaler from: {SCALER_PATH}")

        try:
            # Load saved model
            model = joblib.load(
                MODEL_PATH
            )

            # Load saved scaler
            scaler = joblib.load(
                SCALER_PATH
            )

            print(
                "[+] Model and scaler loaded successfully."
            )

            # Calculate population statistics
            calculate_population_stats()

            # Build model metadata
            model_metrics = {
                "model_type": type(model).__name__
            }

            # Add feature importances if available
            if hasattr(
                model,
                "feature_importances_"
            ):
                importances = (
                    model.feature_importances_
                )

                model_metrics[
                    "feature_importances"
                ] = dict(
                    zip(
                        FEATURE_NAMES,
                        [
                            round(
                                float(value),
                                4
                            )
                            for value in importances
                        ]
                    )
                )

            print(
                f"[+] Loaded model type: "
                f"{type(model).__name__}"
            )

        except Exception as e:

            print(
                f"[!] Failed to load saved model: {e}"
            )

            print(
                "[*] Falling back to baseline model..."
            )

            train_default_baseline_model()

    else:

        print(
            "[*] Saved model/scaler not found."
        )

        print(
            "[*] Training baseline model..."
        )

        train_default_baseline_model()


# ============================================================
# INITIALIZE MODEL
# ============================================================

load_or_init_model()


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():
    """
    Render the primary web interface.
    """

    return render_template(
        "index.html"
    )


# ============================================================
# MODEL INFORMATION API
# ============================================================

@app.route(
    "/api/model-info",
    methods=["GET"]
)
def get_model_info():
    """
    Return model metadata, metrics,
    feature labels and population statistics.
    """

    return jsonify({
        "status": "success",

        "metrics": model_metrics,

        "features": FEATURE_LABELS,

        "population_stats": population_stats
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():
    """
    Accept feature values as JSON and return
    malignant/benign prediction and risk information.
    """

    try:

        # ----------------------------------------------------
        # Read JSON request
        # ----------------------------------------------------

        data = request.get_json(
            force=True
        )

        if not data:

            return jsonify({
                "status": "error",
                "message": "No input data provided"
            }), 400


        # ----------------------------------------------------
        # Extract the 10 required features
        # ----------------------------------------------------

        input_values = []
        feature_dict = {}

        for feature in FEATURE_NAMES:

            value = float(
                data.get(
                    feature,
                    0.0
                )
            )

            input_values.append(value)

            feature_dict[
                feature
            ] = value


        # ----------------------------------------------------
        # Convert input to NumPy array
        # ----------------------------------------------------

        input_array = np.array(
            [input_values],
            dtype=float
        )


        # ----------------------------------------------------
        # Apply scaler
        # ----------------------------------------------------

        if scaler is not None:

            input_scaled = scaler.transform(
                input_array
            )

        else:

            input_scaled = input_array


        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        if model is None:

            return jsonify({
                "status": "error",
                "message": "Model is not initialized"
            }), 500


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    input_scaled
                )[0]
            )

            # Application convention:
            # index 1 = malignant
            malignant_probability = (
                float(probabilities[1])
                if len(probabilities) > 1
                else float(probabilities[0])
            )

        else:

            prediction_raw = int(
                model.predict(
                    input_scaled
                )[0]
            )

            malignant_probability = (
                0.95
                if prediction_raw == 1
                else 0.05
            )


        benign_probability = (
            1.0 -
            malignant_probability
        )


        # ----------------------------------------------------
        # Determine risk category
        # ----------------------------------------------------

        if malignant_probability >= 0.70:

            risk_level = "High Risk"

            diagnosis = (
                "Malignant (Positive)"
            )

            risk_color = "#FF2E93"

            confidence_tag = (
                "Strong Signal"
            )

        elif malignant_probability >= 0.35:

            risk_level = "Moderate Risk"

            diagnosis = (
                "Inconclusive / Borderline"
            )

            risk_color = "#FFB800"

            confidence_tag = (
                "Requires Further Clinical Review"
            )

        else:

            risk_level = "Low Risk"

            diagnosis = (
                "Benign (Negative)"
            )

            risk_color = "#00E699"

            confidence_tag = (
                "High Confidence Benign"
            )


        # ----------------------------------------------------
        # Identify anomalous features
        # ----------------------------------------------------

        anomalies = []

        benign_means = (
            population_stats.get(
                "benign_means",
                {}
            )
        )

        malignant_means = (
            population_stats.get(
                "malignant_means",
                {}
            )
        )


        for feature in FEATURE_NAMES:

            value = feature_dict[
                feature
            ]

            benign_average = (
                benign_means.get(
                    feature,
                    value
                )
            )

            malignant_average = (
                malignant_means.get(
                    feature,
                    value
                )
            )


            if benign_average > 0:

                deviation_percentage = (
                    (
                        value -
                        benign_average
                    )
                    /
                    benign_average
                ) * 100


                if deviation_percentage > 25:

                    anomalies.append({

                        "feature": feature,

                        "label": FEATURE_LABELS.get(
                            feature,
                            feature
                        ),

                        "user_val": value,

                        "benign_avg": round(
                            benign_average,
                            4
                        ),

                        "malignant_avg": round(
                            malignant_average,
                            4
                        ),

                        "elevation_pct": round(
                            deviation_percentage,
                            1
                        )
                    })


        # ----------------------------------------------------
        # Return prediction response
        # ----------------------------------------------------

        return jsonify({

            "status": "success",

            "prediction": {

                "diagnosis": diagnosis,

                "is_malignant": (
                    malignant_probability >= 0.5
                ),

                "malignant_probability": round(
                    malignant_probability * 100,
                    2
                ),

                "benign_probability": round(
                    benign_probability * 100,
                    2
                ),

                "risk_level": risk_level,

                "risk_color": risk_color,

                "confidence_tag": confidence_tag
            },

            "anomalies": sorted(
                anomalies,
                key=lambda item: item[
                    "elevation_pct"
                ],
                reverse=True
            )[:4],

            "input_features": feature_dict
        })


    except Exception as e:

        print(
            f"[!] Prediction error: {e}"
        )

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500


# ============================================================
# LOCAL DEVELOPMENT SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "  ONCOVISION - BREAST CANCER "
        "PREDICTION WEB SERVER"
    )

    print("=" * 60)

    print(
        "  Running on: http://127.0.0.1:5000"
    )

    print(
        f"  Project Base Directory: {BASE_DIR}"
    )

    print(
        f"  Model Path: {MODEL_PATH}"
    )

    print(
        f"  Scaler Path: {SCALER_PATH}"
    )

    print("=" * 60 + "\n")

    app.run(
        debug=True,
        port=5000
    )