# 🗂️ PROJECT FILE STRUCTURE & GUIDE

## Complete Directory Structure

```
📁 Phishing detection system/
│
├── 🌐 WEB APPLICATION
│   ├── app.py (150 lines)
│   │   └── Flask web server, API endpoints, model loading
│   │
│   ├── 📁 templates/
│   │   ├── index.html (100 lines)
│   │   │   └── Home page with navigation
│   │   ├── url_detector.html (120 lines)
│   │   │   └── URL phishing detection interface
│   │   └── email_detector.html (130 lines)
│   │       └── Email phishing detection interface
│   │
│   └── 📁 static/
│       ├── style.css (400 lines)
│       │   └── Complete styling for all pages
│       ├── url_detector.js (80 lines)
│       │   └── URL detection form handling and AJAX
│       └── email_detector.js (90 lines)
│           └── Email detection form handling and AJAX
│
├── 🧠 MACHINE LEARNING CORE
│   └── 📁 src/
│       ├── __init__.py (20 lines)
│       │   └── Package initialization and exports
│       │
│       ├── feature_extraction.py (350 lines)
│       │   ├── URLFeatureExtractor class
│       │   │   └── 30+ URL features (domain, protocol, entropy, etc.)
│       │   └── EmailFeatureExtractor class
│       │       └── 25+ Email features (subject, body, sender, HTML)
│       │
│       ├── url_classifier.py (250 lines)
│       │   └── URLPhishingClassifier class
│       │       ├── Train model
│       │       ├── Make predictions
│       │       ├── Save/load models
│       │       └── Batch processing
│       │
│       ├── email_classifier.py (250 lines)
│       │   └── EmailPhishingClassifier class
│       │       ├── Train model
│       │       ├── Make predictions
│       │       ├── Save/load models
│       │       └── Batch processing
│       │
│       ├── train.py (200 lines)
│       │   ├── Generate sample data
│       │   ├── Train URL classifier
│       │   ├── Train email classifier
│       │   └── Save trained models
│       │
│       └── predict.py (200 lines)
│           ├── Command-line interface
│           ├── Interactive mode
│           ├── Direct prediction mode
│           └── Result formatting
│
├── 📊 DATA & MODELS
│   ├── 📁 data/
│   │   ├── sample_urls.csv (30 samples)
│   │   │   └── Legitimate and phishing URLs
│   │   └── sample_emails.txt (20 samples)
│   │       └── Legitimate and phishing emails
│   │
│   └── 📁 models/ (generated after training)
│       ├── url_classifier.pkl
│       │   └── Trained Random Forest model for URLs
│       └── email_classifier.pkl
│           └── Trained Random Forest model for emails
│
├── 📚 DOCUMENTATION (5 comprehensive guides)
│   ├── README.md (500+ lines)
│   │   ├── Project overview
│   │   ├── Installation guide
│   │   ├── Usage instructions
│   │   ├── API documentation
│   │   ├── Examples
│   │   └── Troubleshooting
│   │
│   ├── INSTALLATION.md (300+ lines)
│   │   ├── Step-by-step installation
│   │   ├── Complete workflow examples
│   │   ├── Testing guide
│   │   └── Detailed troubleshooting
│   │
│   ├── QUICKSTART.md (150+ lines)
│   │   ├── Quick installation
│   │   ├── Usage options
│   │   ├── Common issues
│   │   └── Testing examples
│   │
│   ├── PROJECT_OVERVIEW.md (400+ lines)
│   │   ├── Executive summary
│   │   ├── Technical architecture
│   │   ├── Component details
│   │   ├── Performance metrics
│   │   ├── Deployment options
│   │   └── Customization guide
│   │
│   └── PROJECT_SUMMARY.md (300+ lines)
│       ├── Completion summary
│       ├── File structure
│       ├── Usage guide
│       ├── Performance expectations
│       └── Next steps
│
├── 🛠️ CONFIGURATION & SETUP
│   ├── requirements.txt (15 lines)
│   │   └── All Python dependencies with versions
│   │
│   ├── setup.py (150 lines)
│   │   ├── Automated setup script
│   │   ├── Dependency checking
│   │   ├── Model training
│   │   └── System testing
│   │
│   └── .gitignore (50 lines)
│       └── Git ignore rules for Python, models, IDE files
│
└── 📋 THIS FILE
    └── PROJECT_STRUCTURE.md
        └── Visual guide to all files and their purposes
```

---

## 📊 Statistics

| Category | Count | Lines of Code |
|----------|-------|---------------|
| **Python Files** | 6 | ~1,400 |
| **HTML Files** | 3 | ~350 |
| **CSS Files** | 1 | ~400 |
| **JavaScript Files** | 2 | ~170 |
| **Documentation** | 5 | ~1,650 |
| **Configuration** | 2 | ~165 |
| **Data Files** | 2 | ~50 |
| **TOTAL** | **21** | **~4,185** |

---

## 🎯 File Purposes Quick Reference

### Core Application Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `app.py` | Main web server | Routes, API endpoints, model loading |
| `src/feature_extraction.py` | Feature engineering | Extract features from URLs and emails |
| `src/url_classifier.py` | URL ML model | Train, predict, save/load URL classifier |
| `src/email_classifier.py` | Email ML model | Train, predict, save/load email classifier |
| `src/train.py` | Training pipeline | Automated model training |
| `src/predict.py` | Prediction interface | CLI for making predictions |

### Web Interface Files

| File | Purpose | Components |
|------|---------|------------|
| `templates/index.html` | Home page | Welcome, navigation cards |
| `templates/url_detector.html` | URL checker | URL input form, results display |
| `templates/email_detector.html` | Email checker | Email form, results display |
| `static/style.css` | Styling | Complete UI styling |
| `static/url_detector.js` | URL detection JS | Form handling, AJAX calls |
| `static/email_detector.js` | Email detection JS | Form handling, AJAX calls |

### Documentation Files

| File | Best For | Content |
|------|----------|---------|
| `README.md` | Comprehensive reference | Everything about the project |
| `INSTALLATION.md` | First-time setup | Detailed installation steps |
| `QUICKSTART.md` | Quick answers | Fast reference guide |
| `PROJECT_OVERVIEW.md` | Technical details | Architecture and internals |
| `PROJECT_SUMMARY.md` | Project overview | What's included, how to use |

---

## 🔍 Code Organization

### Feature Extraction Module
```python
src/feature_extraction.py
├── URLFeatureExtractor
│   ├── __init__(): Initialize suspicious patterns
│   ├── extract_features(url): Main feature extraction
│   └── _calculate_entropy(text): Entropy calculation
│
└── EmailFeatureExtractor
    ├── __init__(): Initialize phishing keywords
    ├── extract_features(email_data): Main feature extraction
    └── _calculate_entropy(text): Entropy calculation
```

### Classifier Modules
```python
src/url_classifier.py & src/email_classifier.py
└── PhishingClassifier
    ├── __init__(model_type): Initialize model
    ├── prepare_features(data): Process input data
    ├── train(data, labels): Train the model
    ├── predict(input): Make single prediction
    ├── predict_batch(inputs): Batch predictions
    ├── save_model(path): Save trained model
    └── load_model(path): Load trained model
```

### Web Application
```python
app.py
├── load_models(): Load trained classifiers
├── Routes:
│   ├── /: Home page
│   ├── /url-detector: URL detection page
│   └── /email-detector: Email detection page
├── API Endpoints:
│   ├── /api/check-url: URL detection API
│   ├── /api/check-email: Email detection API
│   └── /api/health: Health check
└── Helper Functions:
    └── get_risk_level(): Calculate risk level
```

---

## 🚀 Execution Flow

### URL Detection Flow

```
User Input (URL)
    ↓
[Web Form or CLI]
    ↓
app.py or predict.py
    ↓
URLFeatureExtractor.extract_features()
    ├── Analyze domain
    ├── Check protocol
    ├── Count special characters
    ├── Calculate entropy
    └── Detect suspicious patterns
    ↓
[30+ features extracted]
    ↓
StandardScaler.transform()
    ↓
RandomForestClassifier.predict()
    ↓
[Prediction + Confidence]
    ↓
Display Results
    ├── Phishing/Legitimate
    ├── Confidence score
    ├── Risk level
    └── Recommendations
```

### Email Detection Flow

```
User Input (Email)
    ↓
[Web Form or CLI]
    ↓
app.py or predict.py
    ↓
EmailFeatureExtractor.extract_features()
    ├── Analyze subject
    ├── Parse body
    ├── Check sender
    ├── Examine HTML
    └── Detect phishing keywords
    ↓
[25+ features extracted]
    ↓
StandardScaler.transform()
    ↓
RandomForestClassifier.predict()
    ↓
[Prediction + Confidence]
    ↓
Display Results
    ├── Phishing/Legitimate
    ├── Confidence score
    ├── Risk level
    └── Security advice
```

### Training Flow

```
src/train.py
    ↓
Generate Sample Data
    ├── Legitimate URLs/Emails
    └── Phishing URLs/Emails
    ↓
URLPhishingClassifier.train()
    ├── Extract features
    ├── Split train/test
    ├── Scale features
    ├── Train Random Forest
    ├── Evaluate performance
    └── Save model
    ↓
EmailPhishingClassifier.train()
    ├── Extract features
    ├── Split train/test
    ├── Scale features
    ├── Train Random Forest
    ├── Evaluate performance
    └── Save model
    ↓
Display Metrics
    ├── Accuracy
    ├── Precision
    ├── Recall
    ├── F1-Score
    └── Confusion Matrix
```

---

## 🎨 UI Components

### Home Page (`templates/index.html`)
- Header with title
- Introduction section
- Two navigation cards:
  - URL Detection card
  - Email Detection card
- Information about phishing
- How it works section
- Footer

### URL Detector Page (`templates/url_detector.html`)
- Header with back link
- URL input form
- Analyze button
- Results section (hidden initially):
  - URL display
  - Prediction badge
  - Confidence bar
  - Risk level
  - Warning/Safe message
- Loading spinner
- Tips section

### Email Detector Page (`templates/email_detector.html`)
- Header with back link
- Email input form:
  - Sender field
  - Subject field
  - Body textarea
- Analyze button
- Results section (hidden initially):
  - Email info display
  - Prediction badge
  - Confidence bar
  - Risk level
  - Warning/Safe message
- Loading spinner
- Tips section

---

## 📦 Dependencies Explained

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | 3.0.0 | Web framework for the application |
| pandas | 2.1.4 | Data manipulation and feature DataFrames |
| numpy | 1.26.2 | Numerical operations and arrays |
| scikit-learn | 1.3.2 | Machine learning models and metrics |
| joblib | 1.3.2 | Model serialization (save/load) |
| tldextract | 5.1.1 | Extract domain parts from URLs |
| requests | 2.31.0 | HTTP requests (optional) |
| beautifulsoup4 | 4.12.2 | HTML parsing (optional) |

---

## 🔧 Configuration Points

### Model Configuration
**Location:** `src/url_classifier.py` and `src/email_classifier.py`

```python
# Change model type
classifier = URLPhishingClassifier(model_type='random_forest')
# Options: 'random_forest', 'gradient_boosting', 'logistic_regression'

# Adjust Random Forest parameters
RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples per leaf
    random_state=42
)
```

### Feature Configuration
**Location:** `src/feature_extraction.py`

```python
# Add suspicious words for URL detection
self.suspicious_words = [
    'secure', 'account', 'update', 'login',
    # Add your words here
]

# Add phishing keywords for email detection
self.phishing_keywords = [
    'urgent', 'verify', 'suspended', 'click here',
    # Add your keywords here
]
```

### Web Server Configuration
**Location:** `app.py`

```python
# Change port and host
app.run(
    debug=True,           # Set to False in production
    host='0.0.0.0',       # Allow external connections
    port=5000             # Change port if needed
)
```

### Training Configuration
**Location:** `src/train.py`

```python
# Adjust train/test split
metrics = classifier.train(urls, labels, test_size=0.25)
# test_size: proportion for testing (0.2 = 20%, 0.25 = 25%, etc.)
```

---

## 🎯 Usage Patterns

### Pattern 1: Web Application User
```
1. python app.py
2. Open http://localhost:5000
3. Click "Check URL" or "Check Email"
4. Enter data
5. Get results
```

### Pattern 2: Command Line User
```
1. cd src
2. python predict.py
3. Choose mode (1 or 2)
4. Enter data
5. Get results
```

### Pattern 3: Developer Integration
```python
# Import classifier
from src.url_classifier import URLPhishingClassifier

# Load model
clf = URLPhishingClassifier()
clf.load_model('models/url_classifier.pkl')

# Integrate into your code
def check_url(url):
    prediction, confidence = clf.predict(url)
    return {'is_phishing': bool(prediction), 'confidence': confidence}
```

### Pattern 4: Batch Processing
```python
# Load classifier
clf = URLPhishingClassifier()
clf.load_model('models/url_classifier.pkl')

# Process multiple URLs
urls = ["http://url1.com", "http://url2.com", ...]
results = clf.predict_batch(urls)

# Process results
for url, (pred, conf) in zip(urls, results):
    print(f"{url}: {'Phishing' if pred else 'Safe'} ({conf:.2%})")
```

---

## 💾 Data Storage

### Models
- **Location:** `models/`
- **Format:** Pickle (.pkl)
- **Contents:** 
  - Trained classifier
  - Scaler
  - Feature names
  - Model type
- **Size:** ~1-5 MB each

### Training Data
- **Location:** `data/`
- **Format:** CSV, TXT
- **Contents:** Sample URLs and emails
- **Purpose:** Example data for training

---

## ✅ Quality Checklist

Before using in production:

- [ ] Trained with sufficient data (1000+ samples)
- [ ] Tested with real phishing examples
- [ ] Adjusted thresholds for your use case
- [ ] Added domain-specific features
- [ ] Implemented logging
- [ ] Set up monitoring
- [ ] Added rate limiting (if using API)
- [ ] Implemented authentication (if public)
- [ ] Updated dependencies
- [ ] Configured for production (debug=False)

---

## 🎓 Learning Path

### Beginner
1. Run setup.py
2. Use web interface
3. Try sample URLs/emails
4. Read QUICKSTART.md

### Intermediate
1. Run training script
2. Use command line
3. Modify features
4. Read README.md

### Advanced
1. Implement custom features
2. Train with your data
3. Adjust model parameters
4. Read PROJECT_OVERVIEW.md
5. Integrate with systems

---

**This structure provides a complete, professional phishing detection system ready for use, learning, or further development!** 🚀

