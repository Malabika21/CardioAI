import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def send_ecg_report_to_doctor(doctor_email, patient_name, patient_email, ecg_result, confidence, analysis_date):
    """Send ECG analysis report to doctor via email"""
    
    sender_email = 'ayushtiwari.creatorslab@gmail.com'
    sender_password = 'tecx bcym vxdz dtni'
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Determine urgency based on result
    urgency_class = "urgent" if ecg_result != "Normal" else "normal"
    urgency_color = "#dc2626" if ecg_result != "Normal" else "#10b981"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f9f9f9;
            }}
            .header {{
                background: linear-gradient(135deg, #3b82f6, #06b6d4);
                color: white;
                padding: 30px;
                text-align: center;
                border-radius: 10px 10px 0 0;
            }}
            .content {{
                background: white;
                padding: 30px;
                border-radius: 0 0 10px 10px;
            }}
            .patient-info {{
                background: #dbeafe;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .result-box {{
                background: {urgency_color};
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                text-align: center;
            }}
            .result-box h3 {{
                margin: 0 0 10px 0;
                font-size: 24px;
            }}
            .result-box p {{
                margin: 5px 0;
                font-size: 16px;
            }}
            .info-row {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #e5e7eb;
            }}
            .info-row:last-child {{
                border-bottom: none;
            }}
            .footer {{
                text-align: center;
                margin-top: 20px;
                color: #666;
                font-size: 12px;
            }}
            .button {{
                display: inline-block;
                background: linear-gradient(135deg, #3b82f6, #06b6d4);
                color: white;
                padding: 12px 30px;
                text-decoration: none;
                border-radius: 8px;
                margin-top: 15px;
            }}
            .urgent-badge {{
                display: inline-block;
                background: #dc2626;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 CardioAI</h1>
                <p>ECG Analysis Report</p>
                {f'<div class="urgent-badge">⚠️ REQUIRES ATTENTION</div>' if ecg_result != "Normal" else ''}
            </div>
            
            <div class="content">
                <h2>ECG Analysis Notification</h2>
                
                <div class="patient-info">
                    <h3 style="margin-top: 0;">Patient Information</h3>
                    <div class="info-row">
                        <strong>Name:</strong>
                        <span>{patient_name}</span>
                    </div>
                    <div class="info-row">
                        <strong>Email:</strong>
                        <span>{patient_email}</span>
                    </div>
                    <div class="info-row">
                        <strong>Analysis Date:</strong>
                        <span>{analysis_date}</span>
                    </div>
                </div>
                
                <div class="result-box">
                    <h3>📊 ECG Result: {ecg_result}</h3>
                    <p>Confidence Score: {confidence:.1f}%</p>
                    <p style="font-size: 14px; margin-top: 10px;">
                        {f'⚠️ This result requires immediate medical review' if ecg_result != "Normal" else '✓ Normal cardiac rhythm detected'}
                    </p>
                </div>
                
                <h3>Analysis Summary</h3>
                <div class="info-row">
                    <strong>Classification:</strong>
                    <span>{ecg_result}</span>
                </div>
                <div class="info-row">
                    <strong>AI Confidence:</strong>
                    <span>{confidence:.1f}%</span>
                </div>
                <div class="info-row">
                    <strong>Sent:</strong>
                    <span>{current_date} at {current_time}</span>
                </div>
                
                <p style="margin-top: 20px;">
                    The patient has shared their ECG analysis results with you. 
                    Please review the complete analysis and contact the patient if necessary.
                </p>
                
                <div style="text-align: center;">
                    <a href="http://127.0.0.1:5000/doctor_dashboard" class="button">
                        View in Dashboard →
                    </a>
                </div>
            </div>
            
            <div class="footer">
                <p>This is an automated email from CardioAI ECG Analysis System</p>
                <p>© 2025 CardioAI. All rights reserved.</p>
                <p style="margin-top: 10px; font-size: 11px;">
                    This email contains confidential patient health information.
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    plain_body = f"""
CardioAI - ECG Analysis Report

{'⚠️ URGENT - Requires Medical Review' if ecg_result != "Normal" else 'ECG Analysis Notification'}

Patient Information:
-------------------
Name: {patient_name}
Email: {patient_email}
Analysis Date: {analysis_date}

ECG Analysis Results:
--------------------
Result: {ecg_result}
Confidence: {confidence:.1f}%
Status: {'Abnormal - Requires Review' if ecg_result != "Normal" else 'Normal'}

The patient has shared their ECG analysis results with you for review.

Login to dashboard: http://127.0.0.1:5000/doctor_dashboard

---
This is an automated email from CardioAI
© 2025 CardioAI. All rights reserved.
Confidential Patient Health Information
    """
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = sender_email
        msg['To'] = doctor_email
        msg['Subject'] = f'{"🚨 URGENT " if ecg_result != "Normal" else ""}ECG Report - {patient_name} ({ecg_result})'
        
        part1 = MIMEText(plain_body, 'plain')
        part2 = MIMEText(html_body, 'html')
        
        msg.attach(part1)
        msg.attach(part2)
        
        print(f"Connecting to {smtp_server}...")
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            print("TLS connection established")
            
            server.login(sender_email, sender_password)
            print("Login successful")
            
            server.send_message(msg)
            print(f"Email sent successfully to {doctor_email}")
            
        return True, "Report shared successfully with doctor!"
        
    except Exception as e:
        print(f"Email send failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, f"Failed to send email: {str(e)}"
