from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
import sqlite3
import bcrypt
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime, timedelta
import io
import base64
import shutil
import csv
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from email_helper import send_ecg_report_to_doctor

app = Flask(__name__)
app.secret_key = 'cardioai_secret_key_2024'

# Load trained model and scaler
try:
    model = tf.keras.models.load_model('models/cardiac_arrhythmia_cnn_gru_model.h5')
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']

# Default admin credentials
ADMIN_EMAIL = "admin@cardioai.com"
ADMIN_PASSWORD = "admin123"

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def convert_row_to_dict(row):
    """Convert SQLite Row to dict and parse datetime fields"""
    if row is None:
        return None
    row_dict = dict(row)
    # Parse created_at field if it exists
    if 'created_at' in row_dict and row_dict['created_at']:
        try:
            row_dict['created_at'] = datetime.strptime(row_dict['created_at'], '%Y-%m-%d %H:%M:%S')
        except:
            row_dict['created_at'] = datetime.now()
    return row_dict

def convert_rows_to_dicts(rows):
    """Convert list of SQLite Rows to list of dicts with parsed datetimes"""
    return [convert_row_to_dict(row) for row in rows]

def init_db():
    conn = get_db_connection()
    
    # Users table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            is_blocked INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Patients table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            user_id INTEGER PRIMARY KEY,
            age INTEGER,
            gender TEXT,
            medical_history TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Doctors table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS doctors (
            user_id INTEGER PRIMARY KEY,
            specialization TEXT,
            license_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Predictions table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            filename TEXT,
            result TEXT,
            confidence REAL,
            all_probabilities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users (id)
        )
    ''')
    
    # Prescriptions table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS prescriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            doctor_id INTEGER,
            medication TEXT,
            dosage TEXT,
            frequency TEXT,
            duration TEXT,
            instructions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users (id),
            FOREIGN KEY (doctor_id) REFERENCES users (id)
        )
    ''')
    
    # System logs table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_type TEXT,
            description TEXT,
            user_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def log_activity(log_type, description, user_id=None):
    """Log system activity"""
    try:
        conn = get_db_connection()
        conn.execute('INSERT INTO system_logs (log_type, description, user_id) VALUES (?, ?, ?)',
                    (log_type, description, user_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging activity: {e}")

def predict_arrhythmia(csv_file_path):
    try:
        # Read CSV file (explicitly ignore any index columns)
        data = pd.read_csv(csv_file_path, header=None, index_col=False)
        
        # If there's an extra column at the beginning (like an index), drop it
        if data.shape[1] == 188:
            data = data.iloc[:, 1:]  # Drop the first column
            print(f"Dropped first column, now have {data.shape[1]} features")
        
        # Validate 187 features
        if data.shape[1] != 187:
            return None, f"Invalid file format. Expected 187 features, got {data.shape[1]}"
        
        # Get first row
        ecg_data = data.iloc[0].values.reshape(1, -1)
        
        # Scale data
        ecg_scaled = scaler.transform(ecg_data)
        
        # Reshape for CNN-GRU
        ecg_final = ecg_scaled.reshape(1, 187, 1)
        
        # Predict
        predictions = model.predict(ecg_final, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        
        return {
            'class': class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': ','.join([str(round(p*100, 2)) for p in predictions]),
            'ecg_data': ecg_data.flatten()
        }, None
        
    except Exception as e:
        return None, str(e)

def generate_ecg_plot(ecg_data, prediction_result):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_data, linewidth=2, color='blue')
        plt.title(f'ECG Analysis: {prediction_result["class"]} ({prediction_result["confidence"]:.1f}% confidence)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_filename = f'ecg_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plot_path = os.path.join('static', 'plots', plot_filename)
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return f'/static/plots/{plot_filename}'
    except Exception as e:
        print(f"Error generating plot: {e}")
        return None

# Decorators for role-based access
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def patient_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'patient':
            flash('Access denied. Patient login required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def doctor_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'doctor':
            flash('Access denied. Doctor login required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def admin_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash('Access denied. Admin login required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        
        # Check for admin login
        if role == 'admin' and email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session['user_id'] = 0
            session['name'] = 'Administrator'
            session['email'] = ADMIN_EMAIL
            session['role'] = 'admin'
            log_activity('login', 'Admin logged in', 0)
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        
        # Regular user login
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ? AND role = ?', (email, role)).fetchone()
        conn.close()
        
        if user:
            # Check if user is blocked
            if user['is_blocked'] == 1:
                flash('Your account has been blocked. Please contact admin.', 'error')
                return redirect(url_for('login'))
            
            if bcrypt.checkpw(password.encode('utf-8'), user['password']):
                session['user_id'] = user['id']
                session['name'] = user['name']
                session['email'] = user['email']
                session['role'] = user['role']
                log_activity('login', f'{user["name"]} logged in', user['id'])
                flash(f'Welcome back, {user["name"]}!', 'success')
                
                if role == 'patient':
                    return redirect(url_for('patient_dashboard'))
                elif role == 'doctor':
                    return redirect(url_for('doctor_dashboard'))
        else:
            flash('Invalid credentials or role mismatch!', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        
        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        try:
            conn = get_db_connection()
            
            # Insert user
            cursor = conn.execute('INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)',
                                (name, email, hashed_password, role))
            user_id = cursor.lastrowid
            
            # Insert role-specific data
            if role == 'patient':
                age = request.form.get('age')
                gender = request.form.get('gender')
                medical_history = request.form.get('medical_history')
                conn.execute('INSERT INTO patients (user_id, age, gender, medical_history) VALUES (?, ?, ?, ?)',
                           (user_id, age, gender, medical_history))
            elif role == 'doctor':
                specialization = request.form.get('specialization')
                license_id = request.form.get('license_id')
                conn.execute('INSERT INTO doctors (user_id, specialization, license_id) VALUES (?, ?, ?)',
                           (user_id, specialization, license_id))
            
            conn.commit()
            conn.close()
            
            log_activity('registration', f'New {role} registered: {name}', user_id)
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
            
        except sqlite3.IntegrityError:
            flash('Email already exists!', 'error')
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    user_name = session.get('name', 'User')
    user_id = session.get('user_id')
    log_activity('logout', f'{user_name} logged out', user_id)
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/patient_dashboard')
@patient_required
def patient_dashboard():
    conn = get_db_connection()
    
    # Get patient predictions
    predictions = conn.execute('''
        SELECT * FROM predictions 
        WHERE patient_id = ? 
        ORDER BY created_at DESC
    ''', (session['user_id'],)).fetchall()
    predictions = convert_rows_to_dicts(predictions)
    
    # Get patient prescriptions
    prescriptions = conn.execute('''
        SELECT p.*, u.name as doctor_name 
        FROM prescriptions p
        JOIN users u ON p.doctor_id = u.id
        WHERE p.patient_id = ?
        ORDER BY p.created_at DESC
        LIMIT 5
    ''', (session['user_id'],)).fetchall()
    prescriptions = convert_rows_to_dicts(prescriptions)
    
    # Get patient info
    patient_info = conn.execute('SELECT * FROM patients WHERE user_id = ?', (session['user_id'],)).fetchone()
    
    conn.close()
    
    return render_template('patient_dashboard.html', 
                         predictions=predictions, 
                         prescriptions=prescriptions,
                         patient_info=patient_info)

@app.route('/detection')
@patient_required
def detection():
    return render_template('detection.html')

@app.route('/upload_ecg', methods=['POST'])
@patient_required
def upload_ecg():
    if 'ecg_file' not in request.files:
        flash('No file selected!', 'error')
        return redirect(url_for('detection'))
    
    file = request.files['ecg_file']
    if file.filename == '':
        flash('No file selected!', 'error')
        return redirect(url_for('detection'))
    
    if file and file.filename.endswith('.csv'):
        # Save uploaded file
        filename = f"ecg_{session['user_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join('static', 'uploads', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        
        # Make prediction
        prediction_result, error = predict_arrhythmia(filepath)
        
        if error:
            flash(f'Prediction error: {error}', 'error')
            return redirect(url_for('detection'))
        
        # Generate ECG plot
        plot_path = generate_ecg_plot(prediction_result['ecg_data'], prediction_result)
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.execute('''
            INSERT INTO predictions (patient_id, filename, result, confidence, all_probabilities)
            VALUES (?, ?, ?, ?, ?)
        ''', (session['user_id'], filename, prediction_result['class'], 
              prediction_result['confidence'], prediction_result['all_probabilities']))
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        log_activity('prediction', f'ECG analysis completed for {session["name"]}', session['user_id'])
        
        return redirect(url_for('results', prediction_id=prediction_id))
    
    else:
        flash('Please upload a valid CSV file!', 'error')
        return redirect(url_for('detection'))

@app.route('/results/<int:prediction_id>')
@login_required
def results(prediction_id):
    conn = get_db_connection()
    prediction = conn.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,)).fetchone()
    conn.close()
    
    if not prediction:
        flash('Prediction not found!', 'error')
        return redirect(url_for('patient_dashboard'))
    
    # Check access rights
    if session['role'] == 'patient' and prediction['patient_id'] != session['user_id']:
        flash('Access denied!', 'error')
        return redirect(url_for('patient_dashboard'))
    
    # Convert prediction to dictionary and parse datetime
    prediction_dict = dict(prediction)
    try:
        # Parse the created_at string to datetime object
        prediction_dict['created_at'] = datetime.strptime(prediction_dict['created_at'], '%Y-%m-%d %H:%M:%S')
    except:
        # If parsing fails, use current datetime
        prediction_dict['created_at'] = datetime.now()
    
    # Generate ECG plot if not exists
    ecg_plot_path = None
    try:
        filepath = os.path.join('static', 'uploads', prediction['filename'])
        if os.path.exists(filepath):
            data = pd.read_csv(filepath, header=None)
            ecg_data = data.iloc[0].values
            prediction_result = {
                'class': prediction['result'],
                'confidence': prediction['confidence']
            }
            ecg_plot_path = generate_ecg_plot(ecg_data, prediction_result)
    except:
        pass
    
    return render_template('results.html', prediction=prediction_dict, ecg_plot_path=ecg_plot_path)

@app.route('/doctor_dashboard')
@doctor_required
def doctor_dashboard():
    conn = get_db_connection()
    
    # Get statistics
    total_patients = conn.execute('SELECT COUNT(*) FROM users WHERE role = "patient"').fetchone()[0]
    total_predictions = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    pending_reviews = conn.execute('SELECT COUNT(*) FROM predictions WHERE result != "Normal"').fetchone()[0]
    total_prescriptions = conn.execute('SELECT COUNT(*) FROM prescriptions WHERE doctor_id = ?', (session['user_id'],)).fetchone()[0]
    
    # Get recent predictions
    recent_predictions = conn.execute('''
        SELECT p.*, u.name as patient_name, u.email as patient_email
        FROM predictions p
        JOIN users u ON p.patient_id = u.id
        ORDER BY p.created_at DESC
        LIMIT 10
    ''').fetchall()
    recent_predictions = convert_rows_to_dicts(recent_predictions)
    
    # Get all patients
    patients = conn.execute('''
        SELECT u.*, pt.age, pt.gender,
               COUNT(pr.id) as prediction_count
        FROM users u
        LEFT JOIN patients pt ON u.id = pt.user_id
        LEFT JOIN predictions pr ON u.id = pr.patient_id
        WHERE u.role = "patient"
        GROUP BY u.id
        ORDER BY u.name
    ''').fetchall()
    patients = convert_rows_to_dicts(patients)
    
    # Get abnormal cases
    abnormal_cases = conn.execute('''
        SELECT p.*, u.name as patient_name
        FROM predictions p
        JOIN users u ON p.patient_id = u.id
        WHERE p.result != "Normal"
        ORDER BY p.created_at DESC
        LIMIT 5
    ''').fetchall()
    abnormal_cases = convert_rows_to_dicts(abnormal_cases)
    
    # Get recent prescriptions
    recent_prescriptions = conn.execute('''
        SELECT p.*, u.name as patient_name
        FROM prescriptions p
        JOIN users u ON p.patient_id = u.id
        WHERE p.doctor_id = ?
        ORDER BY p.created_at DESC
        LIMIT 5
    ''', (session['user_id'],)).fetchall()
    recent_prescriptions = convert_rows_to_dicts(recent_prescriptions)
    
    conn.close()
    
    return render_template('doctor_dashboard.html',
                         total_patients=total_patients,
                         total_predictions=total_predictions,
                         pending_reviews=pending_reviews,
                         total_prescriptions=total_prescriptions,
                         recent_predictions=recent_predictions,
                         patients=patients,
                         abnormal_cases=abnormal_cases,
                         recent_prescriptions=recent_prescriptions)

@app.route('/admin_dashboard')
@admin_required
def admin_dashboard():
    conn = get_db_connection()
    
    # Get statistics
    total_users = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    total_doctors = conn.execute('SELECT COUNT(*) FROM users WHERE role = "doctor"').fetchone()[0]
    total_patients = conn.execute('SELECT COUNT(*) FROM users WHERE role = "patient"').fetchone()[0]
    total_predictions = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    
    # Get all users
    users = conn.execute('SELECT * FROM users ORDER BY created_at DESC').fetchall()
    users = convert_rows_to_dicts(users)
    
    # System statistics
    daily_analyses = conn.execute('''
        SELECT COUNT(*) FROM predictions 
        WHERE date(created_at) = date('now')
    ''').fetchone()[0]
    
    normal_count = conn.execute('SELECT COUNT(*) FROM predictions WHERE result = "Normal"').fetchone()[0]
    abnormal_count = total_predictions - normal_count
    
    normal_results = round((normal_count / total_predictions * 100) if total_predictions > 0 else 0, 1)
    abnormal_results = round((abnormal_count / total_predictions * 100) if total_predictions > 0 else 0, 1)
    
    avg_confidence = conn.execute('SELECT AVG(confidence) FROM predictions').fetchone()[0]
    avg_confidence = round(avg_confidence if avg_confidence else 0, 1)
    
    # Recent activities from logs
    recent_activities = conn.execute('''
        SELECT * FROM system_logs 
        ORDER BY created_at DESC 
        LIMIT 10
    ''').fetchall()
    
    # Convert to proper format
    activities = []
    for log in recent_activities:
        activity_type = 'activity'
        if 'registered' in log['description'].lower():
            activity_type = 'registration'
        elif 'ecg' in log['description'].lower() or 'analysis' in log['description'].lower():
            activity_type = 'prediction'
        elif 'prescription' in log['description'].lower():
            activity_type = 'prescription'
        
        activities.append({
            'type': activity_type,
            'description': log['description'],
            'timestamp': datetime.strptime(log['created_at'], '%Y-%m-%d %H:%M:%S') if isinstance(log['created_at'], str) else log['created_at']
        })
    
    conn.close()
    
    return render_template('admin_dashboard.html',
                         total_users=total_users,
                         total_doctors=total_doctors,
                         total_patients=total_patients,
                         total_predictions=total_predictions,
                         users=users,
                         daily_analyses=daily_analyses,
                         normal_results=normal_results,
                         abnormal_results=abnormal_results,
                         avg_confidence=avg_confidence,
                         recent_activities=activities)

# ========== NEW ADMIN ROUTES ==========

@app.route('/user_details/<int:user_id>')
@admin_required
def user_details(user_id):
    """View detailed user information"""
    conn = get_db_connection()
    
    # Get user info
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if not user:
        flash('User not found!', 'error')
        conn.close()
        return redirect(url_for('admin_dashboard'))
    
    user_dict = convert_row_to_dict(user)
    
    # Get role-specific info
    role_info = None
    if user['role'] == 'patient':
        role_info = conn.execute('SELECT * FROM patients WHERE user_id = ?', (user_id,)).fetchone()
        # Get patient predictions
        predictions = conn.execute('''
            SELECT * FROM predictions 
            WHERE patient_id = ? 
            ORDER BY created_at DESC
        ''', (user_id,)).fetchall()
        user_dict['predictions'] = convert_rows_to_dicts(predictions)
        
        # Get prescriptions
        prescriptions = conn.execute('''
            SELECT p.*, u.name as doctor_name
            FROM prescriptions p
            JOIN users u ON p.doctor_id = u.id
            WHERE p.patient_id = ?
            ORDER BY p.created_at DESC
        ''', (user_id,)).fetchall()
        user_dict['prescriptions'] = convert_rows_to_dicts(prescriptions)
        
    elif user['role'] == 'doctor':
        role_info = conn.execute('SELECT * FROM doctors WHERE user_id = ?', (user_id,)).fetchone()
        # Get doctor prescriptions
        prescriptions = conn.execute('''
            SELECT p.*, u.name as patient_name
            FROM prescriptions p
            JOIN users u ON p.patient_id = u.id
            WHERE p.doctor_id = ?
            ORDER BY p.created_at DESC
        ''', (user_id,)).fetchall()
        user_dict['prescriptions'] = convert_rows_to_dicts(prescriptions)
    
    user_dict['role_info'] = role_info
    
    conn.close()
    
    return render_template('user_details.html', user=user_dict)

@app.route('/block_user/<int:user_id>', methods=['POST'])
@admin_required
def block_user(user_id):
    """Block or unblock a user"""
    conn = get_db_connection()
    
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if not user:
        conn.close()
        return jsonify({'success': False, 'message': 'User not found'}), 404
    
    # Toggle block status
    new_status = 0 if user['is_blocked'] == 1 else 1
    conn.execute('UPDATE users SET is_blocked = ? WHERE id = ?', (new_status, user_id))
    conn.commit()
    
    action = 'blocked' if new_status == 1 else 'unblocked'
    log_activity('user_management', f'User {user["name"]} {action} by admin', session['user_id'])
    
    conn.close()
    
    flash(f'User {action} successfully!', 'success')
    return jsonify({'success': True, 'message': f'User {action} successfully', 'blocked': new_status == 1})

@app.route('/export_data')
@admin_required
def export_data():
    """Export all system data to CSV files in a ZIP"""
    try:
        conn = get_db_connection()
        
        # Create temporary directory for CSV files
        export_dir = 'temp_export'
        os.makedirs(export_dir, exist_ok=True)
        
        # Export users
        users = conn.execute('SELECT id, name, email, role, created_at FROM users').fetchall()
        with open(f'{export_dir}/users.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Name', 'Email', 'Role', 'Created At'])
            for user in users:
                writer.writerow(user)
        
        # Export predictions
        predictions = conn.execute('''
            SELECT p.id, u.name as patient_name, p.result, p.confidence, p.created_at
            FROM predictions p
            JOIN users u ON p.patient_id = u.id
        ''').fetchall()
        with open(f'{export_dir}/predictions.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Patient Name', 'Result', 'Confidence', 'Created At'])
            for pred in predictions:
                writer.writerow(pred)
        
        # Export prescriptions
        prescriptions = conn.execute('''
            SELECT pr.id, p.name as patient_name, d.name as doctor_name, 
                   pr.medication, pr.dosage, pr.created_at
            FROM prescriptions pr
            JOIN users p ON pr.patient_id = p.id
            JOIN users d ON pr.doctor_id = d.id
        ''').fetchall()
        with open(f'{export_dir}/prescriptions.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Patient Name', 'Doctor Name', 'Medication', 'Dosage', 'Created At'])
            for pres in prescriptions:
                writer.writerow(pres)
        
        # Export system logs
        logs = conn.execute('SELECT * FROM system_logs ORDER BY created_at DESC LIMIT 1000').fetchall()
        with open(f'{export_dir}/system_logs.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Log Type', 'Description', 'User ID', 'Created At'])
            for log in logs:
                writer.writerow(log)
        
        conn.close()
        
        # Create ZIP file
        zip_filename = f'cardioai_data_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        shutil.make_archive(f'temp_export/{zip_filename}', 'zip', export_dir)
        
        log_activity('data_export', 'System data exported by admin', session['user_id'])
        
        # Send file and cleanup
        response = send_file(f'temp_export/{zip_filename}.zip', 
                           as_attachment=True, 
                           download_name=f'{zip_filename}.zip')
        
        # Schedule cleanup (in production, use a background task)
        try:
            shutil.rmtree(export_dir)
        except:
            pass
        
        return response
        
    except Exception as e:
        flash(f'Error exporting data: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))

@app.route('/backup_system', methods=['POST'])
@admin_required
def backup_system():
    """Create database backup"""
    try:
        backup_dir = 'backups'
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_filename = f'database_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy database file
        shutil.copy2('database.db', backup_path)
        
        log_activity('backup', 'Database backup created by admin', session['user_id'])
        
        return jsonify({
            'success': True, 
            'message': f'Backup created successfully: {backup_filename}',
            'filename': backup_filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Backup failed: {str(e)}'}), 500

@app.route('/generate_system_report')
@admin_required
def generate_system_report():
    """Generate comprehensive system report PDF"""
    try:
        conn = get_db_connection()
        
        # Gather statistics
        total_users = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
        total_doctors = conn.execute('SELECT COUNT(*) FROM users WHERE role = "doctor"').fetchone()[0]
        total_patients = conn.execute('SELECT COUNT(*) FROM users WHERE role = "patient"').fetchone()[0]
        total_predictions = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        total_prescriptions = conn.execute('SELECT COUNT(*) FROM prescriptions').fetchone()[0]
        
        # Analysis statistics
        normal_count = conn.execute('SELECT COUNT(*) FROM predictions WHERE result = "Normal"').fetchone()[0]
        abnormal_count = total_predictions - normal_count
        
        # Recent activities
        recent_users = conn.execute('''
            SELECT COUNT(*) FROM users 
            WHERE created_at >= datetime('now', '-7 days')
        ''').fetchone()[0]
        
        recent_predictions = conn.execute('''
            SELECT COUNT(*) FROM predictions 
            WHERE created_at >= datetime('now', '-7 days')
        ''').fetchone()[0]
        
        conn.close()
        
        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Title style
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        )
        
        # Title
        story.append(Paragraph("CardioAI System Report", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
        story.append(Spacer(1, 0.5*inch))
        
        # System Overview
        story.append(Paragraph("System Overview", heading_style))
        overview_data = [
            ['Metric', 'Count'],
            ['Total Users', str(total_users)],
            ['Doctors', str(total_doctors)],
            ['Patients', str(total_patients)],
            ['Total ECG Analyses', str(total_predictions)],
            ['Total Prescriptions', str(total_prescriptions)],
        ]
        
        overview_table = Table(overview_data, colWidths=[3*inch, 2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#93c5fd')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f9ff')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Analysis Results
        story.append(Paragraph("ECG Analysis Results", heading_style))
        if total_predictions > 0:
            normal_pct = round((normal_count / total_predictions) * 100, 1)
            abnormal_pct = round((abnormal_count / total_predictions) * 100, 1)
        else:
            normal_pct = abnormal_pct = 0
            
        results_data = [
            ['Result Type', 'Count', 'Percentage'],
            ['Normal', str(normal_count), f'{normal_pct}%'],
            ['Abnormal', str(abnormal_count), f'{abnormal_pct}%'],
            ['Total', str(total_predictions), '100%'],
        ]
        
        results_table = Table(results_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#93c5fd')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f9ff')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(results_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Recent Activity
        story.append(Paragraph("Recent Activity (Last 7 Days)", heading_style))
        activity_data = [
            ['Activity', 'Count'],
            ['New User Registrations', str(recent_users)],
            ['ECG Analyses Performed', str(recent_predictions)],
        ]
        
        activity_table = Table(activity_data, colWidths=[3*inch, 2*inch])
        activity_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#93c5fd')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f9ff')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(activity_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER
        )
        story.append(Paragraph("This report is confidential and for administrative use only.", footer_style))
        story.append(Paragraph("CardioAI - AI-Powered Cardiac Arrhythmia Detection System", footer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        log_activity('report_generation', 'System report generated by admin', session['user_id'])
        
        filename = f"CardioAI_System_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')
        
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))

@app.route('/clear_logs', methods=['POST'])
@admin_required
def clear_logs():
    """Clear system logs"""
    try:
        conn = get_db_connection()
        
        # Get count before deletion
        log_count = conn.execute('SELECT COUNT(*) FROM system_logs').fetchone()[0]
        
        # Delete old logs (keep last 100)
        conn.execute('''
            DELETE FROM system_logs 
            WHERE id NOT IN (
                SELECT id FROM system_logs 
                ORDER BY created_at DESC 
                LIMIT 100
            )
        ''')
        conn.commit()
        
        deleted_count = log_count - 100 if log_count > 100 else 0
        
        log_activity('system_maintenance', f'System logs cleared by admin ({deleted_count} logs removed)', session['user_id'])
        
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': f'Logs cleared successfully. {deleted_count} old logs removed.',
            'deleted': deleted_count
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error clearing logs: {str(e)}'}), 500

@app.route('/admin/ecg_analyses')
@admin_required
def admin_ecg_analyses():
    """View all ECG analyses"""
    conn = get_db_connection()
    
    # Get all predictions with patient info
    predictions = conn.execute('''
        SELECT p.*, u.name as patient_name, u.email as patient_email
        FROM predictions p
        JOIN users u ON p.patient_id = u.id
        ORDER BY p.created_at DESC
    ''').fetchall()
    predictions = convert_rows_to_dicts(predictions)
    
    # Statistics
    total = len(predictions)
    normal = len([p for p in predictions if p['result'] == 'Normal'])
    abnormal = total - normal
    
    conn.close()
    
    return render_template('admin_ecg_analyses.html', 
                         predictions=predictions,
                         total=total,
                         normal=normal,
                         abnormal=abnormal)

@app.route('/admin/alerts')
@admin_required
def admin_alerts():
    """View system alerts and notifications"""
    conn = get_db_connection()
    
    alerts = []
    
    # Check for pending doctor registrations
    doctor_count = conn.execute('SELECT COUNT(*) FROM users WHERE role = "doctor"').fetchone()[0]
    if doctor_count > 0:
        alerts.append({
            'type': 'info',
            'title': 'Doctor Registrations',
            'message': f'{doctor_count} doctors registered in the system',
            'timestamp': datetime.now()
        })
    
    # Check for abnormal ECG results in last 24 hours
    abnormal_count = conn.execute('''
        SELECT COUNT(*) FROM predictions 
        WHERE result != "Normal" 
        AND created_at >= datetime('now', '-1 day')
    ''').fetchone()[0]
    
    if abnormal_count > 0:
        alerts.append({
            'type': 'warning',
            'title': 'Abnormal ECG Results',
            'message': f'{abnormal_count} abnormal ECG results detected in the last 24 hours',
            'timestamp': datetime.now()
        })
    
    # System health check
    total_predictions = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    if total_predictions > 1000:
        alerts.append({
            'type': 'info',
            'title': 'System Milestone',
            'message': f'System has processed {total_predictions} ECG analyses',
            'timestamp': datetime.now()
        })
    
    # Maintenance reminder
    alerts.append({
        'type': 'reminder',
        'title': 'System Maintenance',
        'message': 'Scheduled system maintenance: Next Sunday',
        'timestamp': datetime.now()
    })
    
    conn.close()
    
    return render_template('admin_alerts.html', alerts=alerts)

# ========== EXISTING ROUTES CONTINUE ==========

@app.route('/prescription/<int:patient_id>')
@doctor_required
def prescription_form(patient_id):
    conn = get_db_connection()
    
    patients = conn.execute('SELECT * FROM users WHERE role = "patient" ORDER BY name').fetchall()
    selected_patient = conn.execute('SELECT u.*, p.age, p.gender, p.medical_history FROM users u LEFT JOIN patients p ON u.id = p.user_id WHERE u.id = ?', (patient_id,)).fetchone() if patient_id else None
    
    recent_predictions = []
    if selected_patient:
        recent_predictions = conn.execute('''
            SELECT * FROM predictions 
            WHERE patient_id = ? 
            ORDER BY created_at DESC 
            LIMIT 5
        ''', (patient_id,)).fetchall()
        recent_predictions = convert_rows_to_dicts(recent_predictions)
    
    conn.close()
    
    return render_template('prescription.html',
                         patients=patients,
                         selected_patient=selected_patient,
                         recent_predictions=recent_predictions,
                         current_date=datetime.now().strftime('%Y-%m-%d'))

@app.route('/write_prescription', methods=['POST'])
@doctor_required
def write_prescription():
    patient_id = request.form['patient_id']
    medication = request.form['medication']
    dosage = request.form['dosage']
    frequency = request.form['frequency']
    duration = request.form['duration']
    instructions = request.form['instructions']
    
    conn = get_db_connection()
    
    # Get patient name
    patient = conn.execute('SELECT name FROM users WHERE id = ?', (patient_id,)).fetchone()
    
    conn.execute('''
        INSERT INTO prescriptions (patient_id, doctor_id, medication, dosage, frequency, duration, instructions)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, session['user_id'], medication, dosage, frequency, duration, instructions))
    conn.commit()
    conn.close()
    
    log_activity('prescription', f'Prescription written for {patient["name"]} by Dr. {session["name"]}', session['user_id'])
    
    flash('Prescription written successfully!', 'success')
    return redirect(url_for('doctor_dashboard'))

@app.route('/download_sample')
def download_sample():
    """Download sample ECG CSV file"""
    try:
        sample_file_path = os.path.join('dataset', '2Book2.csv')
        if os.path.exists(sample_file_path):
            return send_file(sample_file_path, as_attachment=True, download_name='sample_ecg_187_features.csv')
        else:
            flash('Sample file not found!', 'error')
            return redirect(request.referrer or url_for('detection'))
    except Exception as e:
        flash(f'Error downloading sample file: {str(e)}', 'error')
        return redirect(request.referrer or url_for('detection'))

@app.route('/profile', methods=['GET', 'POST'])
@patient_required
def profile():
    """View and edit patient profile"""
    conn = get_db_connection()
    
    if request.method == 'POST':
        name = request.form['name']
        age = request.form.get('age')
        gender = request.form.get('gender')
        medical_history = request.form.get('medical_history')
        
        # Update user name
        conn.execute('UPDATE users SET name = ? WHERE id = ?', (name, session['user_id']))
        
        # Update patient info
        conn.execute('''
            UPDATE patients 
            SET age = ?, gender = ?, medical_history = ?
            WHERE user_id = ?
        ''', (age, gender, medical_history, session['user_id']))
        
        conn.commit()
        conn.close()
        
        # Update session name
        session['name'] = name
        
        log_activity('profile_update', f'{name} updated their profile', session['user_id'])
        
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    # GET request
    user = conn.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    user = convert_row_to_dict(user)
    
    patient_info = conn.execute('SELECT * FROM patients WHERE user_id = ?', (session['user_id'],)).fetchone()
    
    analysis_count = conn.execute('SELECT COUNT(*) FROM predictions WHERE patient_id = ?', (session['user_id'],)).fetchone()[0]
    
    conn.close()
    
    return render_template('profile.html', user=user, patient_info=patient_info, analysis_count=analysis_count)

@app.route('/prescription_details/<int:prescription_id>')
@patient_required
def prescription_details(prescription_id):
    """View full prescription details"""
    conn = get_db_connection()
    
    prescription = conn.execute('''
        SELECT p.*, u.name as doctor_name 
        FROM prescriptions p
        JOIN users u ON p.doctor_id = u.id
        WHERE p.id = ? AND p.patient_id = ?
    ''', (prescription_id, session['user_id'])).fetchone()
    
    conn.close()
    
    if not prescription:
        flash('Prescription not found!', 'error')
        return redirect(url_for('patient_dashboard'))
    
    prescription = convert_row_to_dict(prescription)
    doctor_name = prescription['doctor_name']
    
    return render_template('prescription_details.html', prescription=prescription, doctor_name=doctor_name)

@app.route('/download_report/<int:prediction_id>')
@login_required
def download_report(prediction_id):
    """Generate and download PDF report"""
    conn = get_db_connection()
    prediction = conn.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,)).fetchone()
    
    if not prediction:
        flash('Prediction not found!', 'error')
        return redirect(url_for('patient_dashboard'))
    
    # Get patient info
    patient = conn.execute('''
        SELECT u.name, u.email, p.age, p.gender, p.medical_history
        FROM users u
        LEFT JOIN patients p ON u.id = p.user_id
        WHERE u.id = ?
    ''', (prediction['patient_id'],)).fetchone()
    
    conn.close()
    
    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.leading = 14
    
    # Title
    story.append(Paragraph("CardioAI ECG Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')
    story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ['Patient Name:', patient['name']],
        ['Email:', patient['email']],
        ['Age:', str(patient['age']) if patient['age'] else 'Not specified'],
        ['Gender:', patient['gender'].capitalize() if patient['gender'] else 'Not specified'],
        ['Analysis Date:', datetime.strptime(prediction['created_at'], '%Y-%m-%d %H:%M:%S').strftime('%B %d, %Y at %I:%M %p')]
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#dbeafe')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#93c5fd')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # ECG Analysis Results
    story.append(Paragraph("ECG Analysis Results", heading_style))
    
    # Result color based on classification
    result_color = colors.HexColor('#10b981') if prediction['result'] == 'Normal' else colors.HexColor('#dc2626')
    
    result_data = [
        ['Classification:', prediction['result']],
        ['Confidence Score:', f"{prediction['confidence']:.1f}%"],
        ['Status:', 'Normal Cardiac Rhythm' if prediction['result'] == 'Normal' else 'Abnormal - Requires Review']
    ]
    
    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#dbeafe')),
        ('BACKGROUND', (1, 0), (1, 0), result_color),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
        ('TEXTCOLOR', (1, 1), (1, -1), colors.black),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#93c5fd')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Classification Breakdown
    story.append(Paragraph("Classification Probability Breakdown", heading_style))
    probabilities = prediction['all_probabilities'].split(',')
    class_names_list = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
    
    prob_data = [['Classification Type', 'Probability']]
    for i, class_name in enumerate(class_names_list):
        prob_data.append([class_name, f"{float(probabilities[i]):.2f}%"])
    
    prob_table = Table(prob_data, colWidths=[3*inch, 3*inch])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#93c5fd')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f9ff')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Medical Interpretation
    story.append(Paragraph("Medical Interpretation", heading_style))
    
    if prediction['result'] == 'Normal':
        interpretation = "Your ECG shows normal sinus rhythm with no detected arrhythmias. This is a healthy cardiac rhythm pattern."
        recommendation = "Continue regular cardiac monitoring and maintain a healthy lifestyle."
    elif prediction['result'] == 'Supraventricular':
        interpretation = "Supraventricular arrhythmia detected. This originates above the ventricles."
        recommendation = "Consult with a cardiologist for evaluation and treatment options."
    elif prediction['result'] == 'Ventricular':
        interpretation = "Ventricular arrhythmia detected. This originates in the ventricles."
        recommendation = "URGENT: Seek immediate medical attention from a cardiologist."
    elif prediction['result'] == 'Fusion':
        interpretation = "Fusion beat detected. This is a combination of normal and abnormal beats."
        recommendation = "Further cardiac evaluation recommended. Consult with your healthcare provider."
    else:
        interpretation = "Unknown rhythm pattern detected that requires further analysis."
        recommendation = "Professional medical evaluation needed. Consult with a cardiologist."
    
    story.append(Paragraph(interpretation, normal_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"<b>Recommendation:</b> {recommendation}", normal_style))
    story.append(Spacer(1, 0.3*inch))
    
    # ECG Plot if exists
    try:
        filepath = os.path.join('static', 'uploads', prediction['filename'])
        if os.path.exists(filepath):
            data = pd.read_csv(filepath, header=None, index_col=False)
            if data.shape[1] == 188:
                data = data.iloc[:, 1:]
            ecg_data = data.iloc[0].values
            
            # Generate plot
            plt.figure(figsize=(10, 4))
            plt.plot(ecg_data, linewidth=1.5, color='#3b82f6')
            plt.title(f'ECG Signal - {prediction["result"]}', fontsize=14, fontweight='bold')
            plt.xlabel('Time Steps', fontsize=10)
            plt.ylabel('Amplitude', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Save to buffer
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            img_buffer.seek(0)
            
            story.append(Paragraph("ECG Signal Visualization", heading_style))
            img = RLImage(img_buffer, width=6.5*inch, height=2.6*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
    except Exception as e:
        print(f"Error adding ECG plot: {e}")
    
    # Disclaimer
    story.append(Spacer(1, 0.3*inch))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER
    )
    story.append(Paragraph("⚠️ Medical Disclaimer", heading_style))
    story.append(Paragraph(
        "This AI analysis is for informational purposes only and should not replace professional medical advice. "
        "Always consult with qualified healthcare providers for medical diagnosis and treatment decisions. "
        "In case of emergency symptoms, seek immediate medical attention.",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    filename = f"ECG_Report_{patient['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype='application/pdf')

@app.route('/share_with_doctor/<int:prediction_id>')
@patient_required
def share_with_doctor(prediction_id):
    """Share ECG analysis with doctor via email"""
    conn = get_db_connection()
    
    prediction = conn.execute('SELECT * FROM predictions WHERE id = ? AND patient_id = ?', 
                             (prediction_id, session['user_id'])).fetchone()
    
    if not prediction:
        flash('Prediction not found!', 'error')
        return redirect(url_for('patient_dashboard'))
    
    # Get patient info
    patient = conn.execute('SELECT name, email FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    
    conn.close()
    
    # Doctor email (hardcoded for now)
    doctor_email = "ayush.qriocity@gmail.com"
    
    # Parse datetime
    try:
        analysis_date = datetime.strptime(prediction['created_at'], '%Y-%m-%d %H:%M:%S').strftime('%B %d, %Y at %I:%M %p')
    except:
        analysis_date = prediction['created_at']
    
    # Send email
    success, message = send_ecg_report_to_doctor(
        doctor_email=doctor_email,
        patient_name=patient['name'],
        patient_email=patient['email'],
        ecg_result=prediction['result'],
        confidence=prediction['confidence'],
        analysis_date=analysis_date
    )
    
    if success:
        flash(message, 'success')
    else:
        flash(message, 'error')
    
    return redirect(url_for('results', prediction_id=prediction_id))

if __name__ == '__main__':
    init_db()
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/plots', exist_ok=True)
    os.makedirs('backups', exist_ok=True)
    app.run(debug=True)