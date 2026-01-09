# 🚀 COMPLETE INSTALLATION & USAGE GUIDE

## 📋 Prerequisites

Before you begin, ensure you have:
- **Python 3.8+** installed on your system
- **pip** (Python package manager)
- **Command line/Terminal** access
- **Web browser** (for web interface)

## 🔧 Installation Steps

### Step 1: Verify Python Installation

Open your terminal/command prompt and run:

```bash
python --version
```

You should see Python 3.8 or higher. If not, download from [python.org](https://python.org).

### Step 2: Navigate to Project Directory

```bash
cd "d:\Cybersec projects\Phishing detection system"
```

### Step 3: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- scikit-learn (machine learning)
- pandas (data processing)
- numpy (numerical computing)
- tldextract (URL parsing)
- And other required packages

### Step 5: Verify Installation

Run the setup script:

```bash
python setup.py
```

Follow the prompts to:
1. Check dependencies ✓
2. Train models ✓
3. Test the system ✓

## 🎯 Usage Guide

### Option 1: Web Application (Easiest)

**1. Start the web server:**
```bash
python app.py
```

**2. Open your browser and go to:**
```
http://localhost:5000
```

**3. Use the interface:**
- Click "Check URL" to analyze suspicious URLs
- Click "Check Email" to analyze suspicious emails
- Get instant results with confidence scores

**4. Stop the server:**
Press `Ctrl+C` in the terminal

### Option 2: Command Line Interface

**Interactive Mode:**
```bash
cd src
python predict.py
```

Then follow the prompts:
- Choose 1 for URL detection
- Choose 2 for Email detection
- Enter the details when prompted

**Direct URL Check:**
```bash
cd src
python predict.py --mode url --url "http://suspicious-site.com"
```

**Direct Email Check:**
```bash
cd src
python predict.py --mode email \
  --sender "security@suspicious.com" \
  --subject "Urgent Action Required" \
  --body "Your account will be suspended..."
```

### Option 3: Python API

Create a Python script:

```python
# check_url.py
from src.url_classifier import URLPhishingClassifier

# Load the trained model
classifier = URLPhishingClassifier()
classifier.load_model('models/url_classifier.pkl')

# Check a URL
url = "http://example.com"
prediction, confidence = classifier.predict(url)

if prediction == 1:
    print(f"⚠️  PHISHING! Confidence: {confidence:.2%}")
else:
    print(f"✓ Safe. Confidence: {confidence:.2%}")
```

Run it:
```bash
python check_url.py
```

## 📚 Complete Workflow Example

### Training the Models

**1. Navigate to src directory:**
```bash
cd src
```

**2. Run training script:**
```bash
python train.py
```

**3. Wait for training to complete:**
You'll see:
- Feature extraction progress
- Training metrics
- Accuracy scores
- Model saved confirmation

**4. Return to main directory:**
```bash
cd ..
```

### Making Predictions

**Example 1: Check a suspicious URL**

```bash
cd src
python predict.py
```

Choose option 1, then enter:
```
http://secure-paypal-verify.tk/login
```

Result:
```
URL PHISHING DETECTION RESULTS
================================
URL: http://secure-paypal-verify.tk/login

Prediction: PHISHING ⚠️
Confidence: 94.5%

⚠️  WARNING: This URL appears to be a phishing attempt!
Do NOT click on this link or enter any personal information.
```

**Example 2: Check an email**

```bash
cd src
python predict.py
```

Choose option 2, then enter:
```
Sender: security@tempmail.com
Subject: URGENT: Your Account Will Be Suspended
Body: Dear customer, your account has been flagged for suspicious 
activity. Click here to verify immediately or your account will be 
permanently closed: http://verify-account.tk
```

Result:
```
EMAIL PHISHING DETECTION RESULTS
=================================
From: security@tempmail.com
Subject: URGENT: Your Account Will Be Suspended

Prediction: PHISHING ⚠️
Confidence: 97.3%

⚠️  WARNING: This email appears to be a phishing attempt!
Do NOT click any links or provide personal information.
```

## 🌐 Using the Web Interface

### Starting the Web App

```bash
python app.py
```

You'll see:
```
============================================================
          PHISHING DETECTION SYSTEM - WEB APP
============================================================

Loading models...
✓ URL classifier loaded successfully
✓ Email classifier loaded successfully

============================================================
Starting web server...
Access the application at: http://localhost:5000
============================================================
```

### Using URL Detector

1. Go to http://localhost:5000
2. Click "Check URL"
3. Enter URL: `http://suspicious-paypal.tk/login`
4. Click "Analyze URL"
5. View results:
   - Prediction badge (Phishing/Legitimate)
   - Confidence bar
   - Risk level
   - Safety recommendations

### Using Email Detector

1. Go to http://localhost:5000
2. Click "Check Email"
3. Fill in:
   - Sender: `alert@tempmail.com`
   - Subject: `Action Required: Verify Account`
   - Body: `Click here to verify your account...`
4. Click "Analyze Email"
5. View detailed results

## 🔍 Testing with Sample Data

### Safe URLs to Test:
```
https://www.google.com
https://www.microsoft.com
https://github.com
https://www.wikipedia.org
https://www.amazon.com
```

### Suspicious URLs to Test:
```
http://secure-paypal.tk/login
http://verify-account.ml/update
http://free-prize.xyz/claim
http://banking-update.ga/verify
http://192.168.1.1/admin
```

### Legitimate Email Template:
```
Sender: colleague@yourcompany.com
Subject: Meeting Tomorrow
Body: Hi, just a reminder about our team meeting tomorrow at 2 PM. 
Looking forward to seeing everyone!
```

### Phishing Email Template:
```
Sender: security@tempmail123.com
Subject: URGENT: Account Suspended
Body: Dear customer, your account has been suspended due to unusual 
activity. Click here to verify your identity immediately: 
http://verify-now.tk or your account will be permanently closed.
```

## 🛠️ Troubleshooting

### Problem: "Module not found" error

**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "Model file not found"

**Solution:**
```bash
cd src
python train.py
cd ..
```

### Problem: "Port 5000 already in use"

**Solution 1 - Kill the process:**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:5000 | xargs kill -9
```

**Solution 2 - Use different port:**
Edit `app.py` and change:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Problem: Training takes too long

**Solution:**
The sample data is small and should train quickly. If slow:
- Close other programs
- Check CPU usage
- Reduce n_estimators in the classifier

### Problem: Low accuracy

**Solution:**
- Train with more data
- Adjust model parameters
- Add more features
- Use cross-validation

## 📊 Understanding Results

### Confidence Scores

- **90-100%**: Very confident in prediction
- **70-90%**: Confident in prediction
- **50-70%**: Moderate confidence
- **Below 50%**: Low confidence, verify manually

### Risk Levels

- **High**: Strong indicators of phishing
- **Medium**: Some suspicious characteristics
- **Low**: Appears legitimate

### What to Do

**If Phishing Detected:**
1. ❌ Do NOT click any links
2. ❌ Do NOT download attachments
3. ❌ Do NOT provide personal info
4. ✓ Report to IT/security team
5. ✓ Delete the email/close the page
6. ✓ Verify through official channels

**If Legitimate:**
1. ✓ Appears safe to proceed
2. ⚠️  Still verify important requests
3. ⚠️  Check sender if unexpected
4. ⚠️  Use official contact methods for verification

## 🎓 Next Steps

### For Learning:
1. Read `README.md` for detailed documentation
2. Review `PROJECT_OVERVIEW.md` for technical details
3. Examine the code in `src/` folder
4. Experiment with different URLs and emails

### For Development:
1. Add more training data
2. Customize features in `feature_extraction.py`
3. Adjust model parameters
4. Create custom classifiers
5. Integrate with your systems

### For Production:
1. Train with large, real-world dataset
2. Set up monitoring and logging
3. Implement rate limiting
4. Add authentication
5. Deploy to cloud platform
6. Set up CI/CD pipeline

## 📞 Getting Help

### Documentation:
- `README.md` - Complete documentation
- `QUICKSTART.md` - Quick reference
- `PROJECT_OVERVIEW.md` - Technical details

### Code Examples:
- `src/url_classifier.py` - See example usage at bottom
- `src/email_classifier.py` - See example usage at bottom
- `src/feature_extraction.py` - See example usage at bottom

### Sample Data:
- `data/sample_urls.csv` - Sample URLs
- `data/sample_emails.txt` - Sample emails

## ✅ Quick Checklist

Before using the system, ensure:
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Models trained (`python src/train.py`)
- [ ] System tested (`python setup.py`)

You're ready to detect phishing! 🛡️

---

**Need Help?** Check the documentation files or review the code comments for detailed information.

**Found a Bug?** Check troubleshooting section above.

**Want to Contribute?** See README.md for contribution guidelines.
