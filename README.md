CardioAI

An AI-powered cardiovascular health prediction web application built with Python and Flask that uses a hybrid CNN-GRU deep learning architecture to analyze ECG and clinical parameters, generating cardiac arrhythmia predictions in real-time.

📌 Project Overview

CardioAI analyzes patient cardiac parameters and multi-lead data to predict cardiac arrhythmia and cardiovascular risks using deep learning models.
The web application provides an intuitive dashboard for medical professionals and individuals to input diagnostic metrics, view real-time risk predictions, download processed reports, and track diagnostic histories.

Supported Diagnostics:

Cardiac Arrhythmia Detection (CNN-GRU Hybrid Model)

Risk Stratification & Feature Scaling

Patient Report Generation

✨ Features

📊 Interactive Web Interface built with Flask and Jinja2 templates

🤖 Hybrid CNN-GRU Neural Network for time-series cardiac pattern recognition

⚡ Automated Scaler & Feature Normalization pipeline via Scikit-Learn

📄 Automated PDF Report Generation for diagnostic summaries

🔄 Dynamic Model Weight Downloading via GitHub Releases

🔒 Secure User Authentication & Password Hashing (Bcrypt)

💾 SQLite Database integration for patient history logging

🌐 Cloud Deployment optimized for Render Web Services

🧠 Machine Learning Architecture
The core model relies on a deep learning CNN-GRU network trained to extract spatial features from cardiac inputs alongside GRU layers to capture sequential time-series patterns.

Features and Data Pipeline:

Diagnostic clinical parameters & scaled feature arrays

Standard Scaler preprocessing (scaler.pkl)

TensorFlow/Keras inference engine (cardiac_arrhythmia_cnn_gru_model.h5)

The pipeline handles model download dynamically at runtime to maintain low repository footprints and bypass host storage constraints.

📊 Diagnostic & Prediction Logic
The decision pipeline scales incoming patient features before feeding them into the inference engine:

Input parameters are standardized using pre-fitted parameters from scaler.pkl.

The CNN-GRU model evaluates input feature sets to output prediction probabilities.

Threshold values categorize output scores into specific risk profiles (e.g., Normal, Arrhythmia Detected).

Results and clinical visualizers are compiled into downloadable PDF diagnostic reports.

🛠️ Technologies Used

Python 3.11[cite: 2, 3]

Flask

Gunicorn[cite: 2]

TensorFlow / Keras (CPU optimized)[cite: 2]

Scikit-Learn[cite: 2]

Pandas & NumPy[cite: 2]

Matplotlib & Pillow[cite: 2]

ReportLab (PDF Generation)[cite: 2]

SQLite3

HTML5 / CSS3 / Jinja2

📂 Project Structure

Plaintext

CardioAI/
│
├── models/
│   ├── cardiac_arrhythmia_cnn_gru_model.h5
│   └── scaler.pkl
├── static/
├── templates/
├── app.py
├── database.db
├── requirements.txt
├── runtime.txt
└── README.md

⚙️ Installation

Clone the repository:

Bash

git clone https://github.com/Malabika21/CardioAI.git

Move into the project directory:

Bash
cd CardioAI

Install the required dependencies:

Bash

pip install -r requirements.txt

Run the Flask application:

Bash

python app.py

🌐 Deployment
The web application is deployed using Render Web Services with a Python 3.11 environment.

Live Application: https://cardioai-i6el.onrender.com/
