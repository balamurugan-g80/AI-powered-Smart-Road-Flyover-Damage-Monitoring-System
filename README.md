# 🚧 AI-Powered Smart Road & Flyover Damage Monitoring System

An industry-level AI-powered infrastructure monitoring platform that automatically detects road and flyover damages from images, videos, and live camera feeds using YOLOv8. The system performs real-time object detection, severity assessment, maintenance recommendations, analytics, explainable AI, automated email notifications, and report generation to support proactive infrastructure maintenance.

---

## 📌 Overview

Road and flyover damages such as potholes and cracks can lead to accidents, increased maintenance costs, and reduced public safety.

This project uses Artificial Intelligence and Computer Vision to detect infrastructure defects automatically and provide intelligent maintenance support for engineers and government authorities.

---

# 🎯 Objectives

- Detect road and flyover damages in real time
- Perform accurate YOLOv8 object detection
- Display live bounding boxes and annotations
- Analyze damage severity
- Generate maintenance recommendations
- Send automated email alerts
- Store detection history
- Generate PDF reports
- Provide analytics dashboard
- Improve road maintenance decision-making

---

# 🚀 Features

### ✅ AI Detection

- Pothole Detection
- Bridge Crack Detection
- Longitudinal Crack Detection
- Surface Damage Detection

---

### ✅ Input Sources

- Image Upload
- Video Upload
- Live Camera Detection
- Real-time Video Processing

---

### ✅ Live Computer Vision

- Live Bounding Boxes
- Live Object Labels
- Confidence Score
- Real-Time Annotation
- Frame-by-Frame Detection

---

### ✅ Damage Analysis

- Damage Type
- Confidence Percentage
- Severity Classification
- Detection History
- Risk Analysis

---

### ✅ Smart Recommendation Engine

Automatically recommends maintenance actions based on detected damage.

Example

| Damage | Recommended Action |
|---------|--------------------|
| Pothole | Immediate patch repair |
| Bridge Crack | Structural inspection |
| Longitudinal Crack | Crack sealing |
| Surface Damage | Preventive resurfacing |

---

### ✅ Email Notification System

Automatically sends email alerts when new damages are detected.

Email contains

- Damage Type
- Severity
- Confidence
- Timestamp
- Maintenance Recommendation
- Annotated Detection Image

---

### ✅ Explainable AI

- SHAP Explainability
- Feature Importance
- AI Prediction Interpretation

---

### ✅ Analytics Dashboard

- Total Detections
- Damage Distribution
- Severity Distribution
- Detection Timeline
- Historical Analysis

---

### ✅ Report Generation

- PDF Report
- Detection Summary
- Maintenance Report

---

### ✅ Database

Stores

- Detection History
- Damage Type
- Confidence
- Severity
- Timestamp
- Email Logs

---

# 🛣 Supported Damage Classes

- 🕳 Pothole
- 🌉 Bridge Crack
- 🛣 Longitudinal Crack
- ⚠ Surface Damage

---

# 🖥 Technology Stack

## Programming Language

- Python

## Deep Learning

- YOLOv8

## Computer Vision

- OpenCV

## Framework

- Streamlit

## Database

- SQLite

## Explainable AI

- SHAP

## Visualization

- Matplotlib
- Pandas

## Email

- Gmail SMTP

---

# 📂 Project Structure

```
AI-powered-Smart-Road-Flyover-Damage-Monitoring-System

│
├── app.py
├── dashboard.py
├── detection_service.py
├── analytics_service.py
├── recommendation_service.py
├── email_service.py
├── pdf_report_service.py
├── notification_service.py
├── image_utils.py
├── video_utils.py
├── config.py
├── database.py
├── models/
│   └── best.pt
├── requirements.txt
└── README.md
```

---

# ⚙ Installation

Clone Repository

```bash
git clone https://github.com/balamurugan-g80/AI-powered-Smart-Road-Flyover-Damage-Monitoring-System.git
```

Go to Project

```bash
cd AI-powered-Smart-Road-Flyover-Damage-Monitoring-System
```

Install Requirements

```bash
pip install -r requirements.txt
```

Run Application

```bash
streamlit run app.py
```

---

# 📊 Workflow

```
Image / Video / Live Camera
            │
            ▼
      YOLOv8 Detection
            │
            ▼
 Live Bounding Box & Annotation
            │
            ▼
   Damage Classification
            │
            ▼
    Severity Assessment
            │
            ▼
 Maintenance Recommendation
            │
            ▼
 Database Storage
            │
            ▼
  Email Notification
            │
            ▼
 Dashboard & PDF Report
```

---

# 📧 Email Notification Workflow

```
Damage Detected
      │
      ▼
Generate Annotated Image
      │
      ▼
Prepare Email
      │
      ▼
Attach Detection Image
      │
      ▼
Send Email Alert
```

---

# 📈 Future Improvements

- GPS Integration
- Drone-Based Inspection
- Multi-Camera Monitoring
- Cloud Deployment
- Mobile Application
- REST API
- Edge AI Deployment
- Real-Time Traffic Integration

---

# 👨‍💻 Developer

**Balamurugan G**

AI & Machine Learning Enthusiast

---
