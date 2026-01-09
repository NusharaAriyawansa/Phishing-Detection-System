# 🎯 QUICK REFERENCE CARD

## ⚡ Fast Commands

### Setup (First Time)
```bash
pip install -r requirements.txt
cd src && python train.py && cd ..
```

### Run Web App
```bash
python app.py
# Open: http://localhost:5000
```

### Command Line Usage
```bash
# Interactive
cd src && python predict.py

# Direct URL
cd src && python predict.py --mode url --url "http://test.com"

# Direct Email
cd src && python predict.py --mode email --sender "x@y.com" --subject "Test" --body "Text"
```

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `app.py` | Web application |
| `src/train.py` | Train models |
| `src/predict.py` | Make predictions |
| `src/feature_extraction.py` | Feature engineering |
| `README.md` | Full documentation |

---

## 🔧 Quick Customization

### Change Model Type
```python
# In url_classifier.py or email_classifier.py
classifier = URLPhishingClassifier(model_type='gradient_boosting')
# Options: random_forest, gradient_boosting, logistic_regression
```

### Add Phishing Keywords
```python
# In src/feature_extraction.py, EmailFeatureExtractor class
self.phishing_keywords = [
    'urgent', 'verify', 'suspended',
    'YOUR_KEYWORD_HERE'  # Add here
]
```

### Adjust Confidence Threshold
```python
# In app.py, get_risk_level function
if confidence >= 0.8:  # Change this value
    return 'High'
```

---

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| Module not found | `pip install -r requirements.txt` |
| Model not found | `cd src && python train.py` |
| Port in use | Change port in `app.py` line 130 |
| Low accuracy | Train with more data |

---

## 📊 API Endpoints

```
POST /api/check-url
Body: {"url": "http://example.com"}

POST /api/check-email
Body: {
  "sender": "x@y.com",
  "subject": "Test",
  "body": "Content"
}

GET /api/health
```

---

## 🎯 Testing URLs

**Safe:** https://www.google.com  
**Phishing:** http://secure-paypal.tk/login

---

## 📚 Documentation

- **Setup:** INSTALLATION.md
- **Quick Start:** QUICKSTART.md
- **Full Docs:** README.md
- **Overview:** PROJECT_OVERVIEW.md

---

## 💡 Pro Tips

1. Train with 1000+ samples for best accuracy
2. Update models regularly with new data
3. Use web interface for demos
4. Use CLI for automation
5. Check confidence scores, not just predictions

---

## 🚀 Production Checklist

- [ ] Train with real data
- [ ] Test thoroughly
- [ ] Set debug=False in app.py
- [ ] Add authentication
- [ ] Set up logging
- [ ] Monitor performance
- [ ] Update dependencies

---

**Quick Help:** See README.md for detailed documentation
