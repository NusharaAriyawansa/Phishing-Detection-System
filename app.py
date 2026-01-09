"""
Flask Web Application for Phishing Detection System
Provides a web interface for URL and Email phishing detection
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.url_classifier import URLPhishingClassifier
from src.email_classifier import EmailPhishingClassifier

app = Flask(__name__)

# Initialize classifiers
url_classifier = None
email_classifier = None

# Load models
def load_models():
    """Load trained models"""
    global url_classifier, email_classifier
    
    url_model_path = os.path.join('models', 'url_classifier.pkl')
    email_model_path = os.path.join('models', 'email_classifier.pkl')
    
    # Load URL classifier
    try:
        url_classifier = URLPhishingClassifier()
        url_classifier.load_model(url_model_path)
        print("✓ URL classifier loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not load URL classifier: {e}")
        print("   Please train the model first by running: python src/train.py")
    
    # Load Email classifier
    try:
        email_classifier = EmailPhishingClassifier()
        email_classifier.load_model(email_model_path)
        print("✓ Email classifier loaded successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not load Email classifier: {e}")
        print("   Please train the model first by running: python src/train.py")


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/url-detector')
def url_detector():
    """URL detector page"""
    return render_template('url_detector.html')


@app.route('/email-detector')
def email_detector():
    """Email detector page"""
    return render_template('email_detector.html')


@app.route('/api/check-url', methods=['POST'])
def check_url():
    """API endpoint to check URL"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({
                'success': False,
                'error': 'URL is required'
            }), 400
        
        if url_classifier is None:
            return jsonify({
                'success': False,
                'error': 'URL classifier not loaded. Please train the model first.'
            }), 500
        
        # Make prediction
        prediction, confidence = url_classifier.predict(url)
        
        result = {
            'success': True,
            'url': url,
            'is_phishing': bool(prediction),
            'confidence': float(confidence),
            'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
            'risk_level': get_risk_level(confidence, prediction)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/check-email', methods=['POST'])
def check_email():
    """API endpoint to check email"""
    try:
        data = request.get_json()
        
        sender = data.get('sender', '').strip()
        subject = data.get('subject', '').strip()
        body = data.get('body', '').strip()
        html = data.get('html', '').strip()
        
        if not sender or not subject or not body:
            return jsonify({
                'success': False,
                'error': 'Sender, subject, and body are required'
            }), 400
        
        if email_classifier is None:
            return jsonify({
                'success': False,
                'error': 'Email classifier not loaded. Please train the model first.'
            }), 500
        
        # Prepare email data
        email_data = {
            'sender': sender,
            'subject': subject,
            'body': body,
            'html': html
        }
        
        # Make prediction
        prediction, confidence = email_classifier.predict(email_data)
        
        result = {
            'success': True,
            'email': {
                'sender': sender,
                'subject': subject,
                'body_preview': body[:200] + ('...' if len(body) > 200 else '')
            },
            'is_phishing': bool(prediction),
            'confidence': float(confidence),
            'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
            'risk_level': get_risk_level(confidence, prediction)
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def get_risk_level(confidence, prediction):
    """Determine risk level based on confidence and prediction"""
    if prediction == 0:
        return 'Low'
    elif confidence >= 0.8:
        return 'High'
    elif confidence >= 0.6:
        return 'Medium'
    else:
        return 'Low'


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'url_classifier_loaded': url_classifier is not None,
        'email_classifier_loaded': email_classifier is not None
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" " * 10 + "PHISHING DETECTION SYSTEM - WEB APP")
    print("=" * 60)
    print("\nLoading models...")
    
    load_models()
    
    print("\n" + "=" * 60)
    print("Starting web server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
