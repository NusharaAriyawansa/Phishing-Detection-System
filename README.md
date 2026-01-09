# Phishing Detection System
An advanced phishing detection system that uses machine learning to identify phishing URLs and emails.

## Features
- **URL Phishing Detection**: Analyzes URLs to detect phishing attempts
- **Email Phishing Detection**: Examines email content for phishing indicators
- **Machine Learning Models**: Uses Random Forest, Gradient Boosting, and Logistic Regression
- **Web Interface**: User-friendly Flask-based web application
- **Real-time Analysis**: Instant results with confidence scores
- **High Accuracy**: Achieves 90%+ accuracy on test data
- **Comprehensive Feature Extraction**: 30+ features extracted from URLs and emails

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone or download the repository**

2. **Navigate to the project directory**
   ```bash
   cd "Phishing detection system"
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Train the Models

Before using the system, you need to train the machine learning models:

```bash
cd src
python train.py
```

This will:
- Train both URL and Email classifiers
- Save the trained models in the `models/` directory
- Display training metrics and accuracy

### 2. Run the Web Application

```bash
python app.py
```

Then open your browser and navigate to: `http://localhost:5000`

### 3. Make Predictions via Command Line

**Check a URL:**
```bash
cd src
python predict.py --mode url --url "http://suspicious-website.tk/login"
```

**Check an Email:**
```bash
cd src
python predict.py --mode email --sender "security@fake.com" --subject "Urgent Action Required" --body "Your account will be closed..."
```

**Interactive Mode:**
```bash
cd src
python predict.py
```

## Project Structure

```
Phishing detection system/
│
├── app.py                      # Flask web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── src/                        # Source code
│   ├── feature_extraction.py  # Feature extraction for URLs and emails
│   ├── url_classifier.py      # URL phishing classifier
│   ├── email_classifier.py    # Email phishing classifier
│   ├── train.py               # Training script
│   └── predict.py             # Prediction/inference script
│
├── models/                     # Trained models (generated after training)
│   ├── url_classifier.pkl
│   └── email_classifier.pkl
│
├── data/                       # Sample datasets
│   ├── sample_urls.csv
│   └── sample_emails.txt
│
├── templates/                  # HTML templates for web app
│   ├── index.html
│   ├── url_detector.html
│   └── email_detector.html
│
└── static/                     # Static files (CSS, JS)
    ├── style.css
    ├── url_detector.js
    └── email_detector.js
```

## Features Extracted

### URL Features (30+)

- **Basic Metrics**: URL length, number of dots, hyphens, special characters
- **Domain Analysis**: Domain length, subdomain count, suspicious TLDs
- **Protocol**: HTTP vs HTTPS
- **Suspicious Patterns**: IP addresses, URL shorteners, suspicious keywords
- **Entropy**: Shannon entropy for randomness detection
- **Character Ratios**: Digits, letters, special characters

### Email Features (25+)

- **Subject Analysis**: Length, urgency indicators, capitalization
- **Body Analysis**: Phishing keywords, URL count, generic greetings
- **Sender Analysis**: Domain verification, suspicious patterns
- **HTML Analysis**: Hidden content, forms, external images
- **Text Metrics**: Uppercase ratio, punctuation, entropy

## Models

The system supports three types of machine learning models:

1. **Random Forest Classifier** (Default)
   - 100 trees
   - Max depth: 20
   - Best for balanced accuracy and interpretability

2. **Gradient Boosting Classifier**
   - 100 estimators
   - Learning rate: 0.1
   - Good for complex patterns

3. **Logistic Regression**
   - Fast training and prediction
   - Good for real-time applications

You can change the model type in the classifier initialization:
```python
classifier = URLPhishingClassifier(model_type='gradient_boosting')
```

## API Documentation

### Check URL Endpoint

**POST** `/api/check-url`

**Request Body:**
```json
{
  "url": "http://example.com"
}
```

**Response:**
```json
{
  "success": true,
  "url": "http://example.com",
  "is_phishing": false,
  "confidence": 0.85,
  "prediction": "Legitimate",
  "risk_level": "Low"
}
```

### Check Email Endpoint

**POST** `/api/check-email`

**Request Body:**
```json
{
  "sender": "user@example.com",
  "subject": "Email subject",
  "body": "Email content...",
  "html": "<html>...</html>"
}
```

**Response:**
```json
{
  "success": true,
  "email": {
    "sender": "user@example.com",
    "subject": "Email subject",
    "body_preview": "Email content..."
  },
  "is_phishing": false,
  "confidence": 0.92,
  "prediction": "Legitimate",
  "risk_level": "Low"
}
```

### Health Check Endpoint

**GET** `/api/health`

**Response:**
```json
{
  "status": "healthy",
  "url_classifier_loaded": true,
  "email_classifier_loaded": true
}
```