# 🩺 OncoVision AI - Breast Cancer Diagnostic & Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2%2B-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](#license)

**OncoVision AI** is a state-of-the-art, glassmorphic medical web application built with **Flask**, **Python**, **Scikit-Learn**, and **Chart.js**. It enables real-time breast cancer malignancy risk prediction using cytological nuclear cell measurements with interactive sliders, clinical presets, and population radar analytics.

---

## ✨ Features & Highlights

- 🎨 **Medical Glassmorphic Aesthetics**: Sleek dark slate theme (`#090C15`) with vibrant magenta-pink glow accents (`#FF2E93`), backdrop blur cards (`backdrop-filter: blur(18px)`), and floating ribbon micro-animations.
- 🔬 **Interactive Diagnostic Laboratory**: 10 adjustable cell nucleus feature inputs (Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension).
- ⚡ **One-Click Clinical Presets**:
  - `⚠️ Malignant Case`: Pre-loads actual clinical malignant dataset measurements.
  - `✅ Benign Case`: Pre-loads actual clinical benign dataset measurements.
  - `🔄 Baseline`: Resets inputs to mean dataset baselines.
- 📊 **Diagnostic Visual Analytics**:
  - **Radar Projection**: Overlays patient sample values against population benign and malignant baselines.
  - **Feature Importance Chart**: Displays relative feature weights from the classifier.
- 📦 **Seamless Model Backend**: Auto-trains a high-accuracy baseline classifier if `model.pkl` is missing on initial startup so the app works out of the box!

---

## 📁 System Architecture & Directory Structure

```text
breast-cancer-flask-app/
├── .gitignore              # Ignores venv, cache, and IDE configs
├── README.md               # GitHub repository documentation
├── app.py                  # Flask web server & REST API endpoints
├── requirements.txt        # Python package dependencies
├── model/                  # Trained machine learning model weights
│   ├── model.pkl           # Saved model classifier object
│   ├── scaler.pkl          # Feature scaling transformer
│   └── lr.pkl              # Logistic Regression baseline
├── static/
│   ├── css/
│   │   └── style.css       # Glassmorphic dark styling & keyframe animations
│   └── js/
│       └── main.js         # Async API fetch, preset handlers & Chart.js radar charts
└── templates/
    └── index.html          # HTML5 single-page web app interface
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/breast-cancer-flask-app.git
cd breast-cancer-flask-app
```

### 2. Set Up Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask Server
```bash
python app.py
```

### 5. Open Web Browser
Navigate to: **`http://127.0.0.1:5000`**

---

## 🛠️ Tech Stack

- **Backend**: Python 3.x, Flask, Scikit-Learn, NumPy, Pandas, Joblib
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism, CSS Variables, Keyframes), JavaScript (ES6+)
- **Data Visualization**: Chart.js 4.x
- **Typography**: Google Fonts (*Plus Jakarta Sans* & *Outfit*)

---

## ⚠️ Clinical Disclaimer

> **IMPORTANT**: OncoVision AI is an educational demonstration project developed to showcase machine learning integration with web technology. It does NOT provide formal medical diagnosis or replace professional clinical evaluation by qualified healthcare providers.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
