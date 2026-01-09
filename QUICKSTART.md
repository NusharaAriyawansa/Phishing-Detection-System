# QUICK START GUIDE

## Installation & Setup

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Quick Setup (Automated)
```bash
python setup.py
```

This will:
- Check all dependencies
- Train the models
- Test the system

### 3. Manual Setup

**Train Models:**
```bash
cd src
python train.py
cd ..
```

**Test Predictions:**
```bash
cd src
python predict.py
```

## Usage Options

### Option 1: Web Application (Recommended for Beginners)

```bash
python app.py
```

Then open your browser to: http://localhost:5000

Features:
- User-friendly interface
- Real-time URL checking
- Email analysis
- Visual results with confidence scores

### Option 2: Command Line Interface

**Interactive Mode:**
```bash
cd src
python predict.py
```

**Direct URL Check:**
```bash
cd src
python predict.py --mode url --url "http://example.com"
```

**Direct Email Check:**
```bash
cd src
python predict.py --mode email --sender "user@example.com" --subject "Test" --body "Email content here"
```

### Option 3: Python API

```python
# URL Detection
from src.url_classifier import URLPhishingClassifier

classifier = URLPhishingClassifier()
classifier.load_model('models/url_classifier.pkl')
prediction, confidence = classifier.predict("http://test.com")

print(f"Phishing: {prediction == 1}, Confidence: {confidence:.2%}")
```

```python
# Email Detection
from src.email_classifier import EmailPhishingClassifier

classifier = EmailPhishingClassifier()
classifier.load_model('models/email_classifier.pkl')

email = {
    'subject': 'Test',
    'body': 'Test email',
    'sender': 'test@example.com',
    'html': ''
}

prediction, confidence = classifier.predict(email)
print(f"Phishing: {prediction == 1}, Confidence: {confidence:.2%}")
```

## Common Issues

**Issue: Models not found**
- Solution: Run `python src/train.py` first

**Issue: Module not found**
- Solution: Run `pip install -r requirements.txt`

**Issue: Port 5000 already in use**
- Solution: Change port in app.py or close the other application

## Testing Examples

### Safe URLs to Test:
- https://www.google.com
- https://www.microsoft.com
- https://github.com

### Suspicious URLs to Test:
- http://secure-paypal.tk/login
- http://192.168.1.1/verify
- http://free-money.ml/claim

### Legitimate Email Example:
- Sender: colleague@company.com
- Subject: Meeting Tomorrow
- Body: Hi, reminder about our meeting at 2 PM

### Phishing Email Example:
- Sender: security@tempmail.com
- Subject: URGENT: Account Suspended
- Body: Click here to verify your account: http://fake-bank.tk

## Next Steps

1. Train with your own data for better accuracy
2. Customize features in feature_extraction.py
3. Adjust model parameters for your use case
4. Integrate with your existing security infrastructure

## Support

- Check README.md for detailed documentation
- Review examples in the src/ folder
- Test with sample data in data/ folder

---

Happy Phishing Detection! 🛡️
