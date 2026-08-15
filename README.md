# 🩺 OncoVision AI — Breast Cancer Diagnostic & Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2%2B-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0-orange.svg)](https://scikit-learn.org/)

**OncoVision AI** is a glassmorphic medical web application built with **Flask**, **Python**, **Scikit-Learn**, and **Chart.js**. It predicts breast cancer malignancy risk in real time from cytological nuclear cell measurements, using interactive sliders, clinical presets, and population-level radar analytics.

🔗 **Live Demo:** [https://breastcancerdetection-ausd.onrender.com/](https://breastcancerdetection-ausd.onrender.com/)

---

## ✨ Features

- 🎨 **Medical Glassmorphic UI** — dark slate theme with magenta-pink glow accents, blurred glass cards, and subtle animations.
- 🔬 **Interactive Diagnostic Panel** — 10 adjustable cell nucleus features: Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, and Fractal Dimension.
- ⚡ **One-Click Clinical Presets**
  - `⚠️ Malignant Case` — loads sample malignant measurements
  - `✅ Benign Case` — loads sample benign measurements
  - `🔄 Baseline` — resets inputs to dataset mean values
- 📊 **Diagnostic Analytics**
  - Radar chart comparing patient input against benign/malignant population baselines
  - Feature importance chart showing classifier weights
- 📦 **Zero-Setup Model Backend** — automatically trains a baseline classifier if `model.pkl` isn't found, so the app runs out of the box.

---

## 📁 Project Structure

```
breastCancerDetection/
├── app.py                  # Flask server & prediction API endpoints
├── requirements.txt        # Python dependencies
├── model/                  # Trained model artifacts
│   ├── model.pkl           # Saved classifier
│   ├── scaler.pkl          # Feature scaler
│   └── lr.pkl              # Logistic Regression model
├── notebook/                # Model training & experimentation notebooks
├── setup guide/             # Setup instructions / documentation
├── static/
│   ├── css/                 # Glassmorphic styling
│   └── js/                  # Chart.js logic, API calls, preset handlers
└── templates/
    └── index.html           # Web app front end
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Ayu-this-side/breastCancerDetection.git
cd breastCancerDetection
```

### 2. (Optional) Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

### 5. Open in your browser
```
http://127.0.0.1:5000
```

---

## 🧠 Model

The prediction engine uses **Logistic Regression** trained on the Breast Cancer Wisconsin (Diagnostic) dataset, with features scaled before inference. If no pre-trained model is found in `model/`, the app trains one automatically on startup.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Backend | Python, Flask, Scikit-Learn, NumPy, Pandas, Joblib |
| Frontend | HTML5, CSS3 (glassmorphism, custom properties), JavaScript (ES6+) |
| Visualization | Chart.js |
| Fonts | Plus Jakarta Sans, Outfit (Google Fonts) |

---

## ⚠️ Disclaimer

> OncoVision AI is an **educational project** demonstrating machine learning integration with web technology. It is **not** a certified diagnostic tool and should never replace professional medical evaluation.
