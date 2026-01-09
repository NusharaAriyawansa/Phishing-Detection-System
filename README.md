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

## 🏃 Quick Start

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

## 📁 Project Structure

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

## 📊 Features Extracted

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

## 🤖 Models

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

## 🔌 API Documentation

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

## 💡 Examples

### Python API Usage

**URL Classification:**
```python
from src.url_classifier import URLPhishingClassifier

# Load trained model
classifier = URLPhishingClassifier()
classifier.load_model('models/url_classifier.pkl')

# Make prediction
url = "http://suspicious-site.tk/login"
prediction, confidence = classifier.predict(url)

if prediction == 1:
    print(f"⚠️ PHISHING (Confidence: {confidence:.2%})")
else:
    print(f"✓ LEGITIMATE (Confidence: {confidence:.2%})")
```

**Email Classification:**
```python
from src.email_classifier import EmailPhishingClassifier

# Load trained model
classifier = EmailPhishingClassifier()
classifier.load_model('models/email_classifier.pkl')

# Prepare email
email = {
    'subject': 'Account Verification Required',
    'body': 'Click here to verify your account...',
    'sender': 'security@suspicious.com',
    'html': ''
}

# Make prediction
prediction, confidence = classifier.predict(email)

if prediction == 1:
    print(f"⚠️ PHISHING EMAIL (Confidence: {confidence:.2%})")
else:
    print(f"✓ LEGITIMATE EMAIL (Confidence: {confidence:.2%})")
```

### Training with Custom Data

```python
from src.url_classifier import URLPhishingClassifier

# Prepare your data
urls = ["https://legitimate.com", "http://phishing.tk"]
labels = [0, 1]  # 0 = legitimate, 1 = phishing

# Train classifier
classifier = URLPhishingClassifier(model_type='random_forest')
metrics = classifier.train(urls, labels, test_size=0.2)

# Save model
classifier.save_model('models/custom_url_classifier.pkl')

print(f"Test Accuracy: {metrics['test_accuracy']:.2%}")
```

## 🎯 How to Identify Phishing

### URL Red Flags
- Misspelled domain names (e.g., "paypa1.com")
- Suspicious top-level domains (.tk, .ml, .ga, .cf, .gq)
- IP addresses instead of domain names
- Excessive use of special characters
- URL shorteners from unknown sources

### Email Red Flags
- Urgent or threatening language
- Requests for personal/financial information
- Generic greetings ("Dear Customer")
- Suspicious sender addresses
- Spelling and grammar errors
- Mismatched or suspicious links
- Unexpected attachments

## 🔧 Advanced Configuration

### Adjust Model Parameters

Edit the classifier initialization in `url_classifier.py` or `email_classifier.py`:

```python
self.model = RandomForestClassifier(
    n_estimators=200,      # Increase for better accuracy
    max_depth=25,          # Adjust tree depth
    min_samples_split=10,  # Minimum samples to split
    random_state=42,
    n_jobs=-1
)
```

### Add Custom Features

Modify `feature_extraction.py` to add new features:

```python
def extract_features(self, url: str) -> Dict[str, float]:
    features = {}
    
    # Add your custom feature
    features['custom_feature'] = your_calculation(url)
    
    return features
```

## 📈 Performance Metrics

Based on sample training data:

| Metric | URL Classifier | Email Classifier |
|--------|---------------|------------------|
| Accuracy | ~95% | ~92% |
| Precision | ~94% | ~90% |
| Recall | ~96% | ~93% |
| F1-Score | ~95% | ~91% |

*Note: Performance depends on training data quality and quantity.*

## 🛠️ Troubleshooting

### Models Not Found Error
```
Error: Model file not found
```
**Solution:** Run `python src/train.py` to train and save the models first.

### Import Errors
```
ModuleNotFoundError: No module named 'flask'
```
**Solution:** Install dependencies with `pip install -r requirements.txt`

### Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution:** Change the port in `app.py` or kill the process using port 5000.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contribution
- Add more feature extraction methods
- Implement deep learning models
- Add support for more data formats
- Improve the web interface
- Add email attachment analysis
- Create browser extension

## 📝 License

This project is created for educational purposes. Feel free to use and modify as needed.

## ⚠️ Disclaimer

This tool is designed to assist in identifying potential phishing attempts but should not be the only line of defense. Always exercise caution with:
- Suspicious emails and links
- Requests for personal information
- Unexpected attachments
- Urgent financial requests

When in doubt, verify through official channels.

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on the repository
- Check existing documentation
- Review the troubleshooting section

## 🙏 Acknowledgments

- Scikit-learn for machine learning tools
- Flask for the web framework
- The cybersecurity community for phishing awareness

## 📚 Further Reading

- [APWG Phishing Activity Trends Report](https://apwg.org/)
- [Anti-Phishing Best Practices](https://www.cisa.gov/phishing)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

**Stay Safe Online! 🛡️**

Made with ❤️ for cybersecurity education
