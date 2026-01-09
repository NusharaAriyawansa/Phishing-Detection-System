"""
Quick Setup and Test Script
Run this to verify installation and test the system
"""

import sys
import os
import subprocess

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def check_dependencies():
    """Check if all dependencies are installed"""
    print_header("CHECKING DEPENDENCIES")
    
    required_packages = [
        'flask', 'pandas', 'numpy', 'sklearn', 
        'joblib', 'tldextract'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("\nPlease run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies are installed!")
        return True

def run_training():
    """Train the models"""
    print_header("TRAINING MODELS")
    print("\nThis will train both URL and Email classifiers...")
    print("(This may take a minute)\n")
    
    try:
        # Change to src directory and run training
        os.chdir('src')
        result = subprocess.run([sys.executable, 'train.py'], 
                              capture_output=False, text=True)
        os.chdir('..')
        
        if result.returncode == 0:
            print("\n✓ Training completed successfully!")
            return True
        else:
            print("\n✗ Training failed!")
            return False
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        os.chdir('..')
        return False

def test_predictions():
    """Test the prediction functionality"""
    print_header("TESTING PREDICTIONS")
    
    try:
        from src.url_classifier import URLPhishingClassifier
        from src.email_classifier import EmailPhishingClassifier
        
        # Test URL classifier
        print("\nTesting URL Classifier...")
        url_classifier = URLPhishingClassifier()
        url_classifier.load_model('models/url_classifier.pkl')
        
        test_url = "http://secure-paypal-verify.tk/login"
        prediction, confidence = url_classifier.predict(test_url)
        
        print(f"  URL: {test_url}")
        print(f"  Prediction: {'PHISHING' if prediction == 1 else 'LEGITIMATE'}")
        print(f"  Confidence: {confidence:.2%}")
        print("  ✓ URL classifier working!")
        
        # Test email classifier
        print("\nTesting Email Classifier...")
        email_classifier = EmailPhishingClassifier()
        email_classifier.load_model('models/email_classifier.pkl')
        
        test_email = {
            'subject': 'URGENT: Verify Your Account',
            'body': 'Click here to verify: http://fake-site.com',
            'sender': 'security@temp.com',
            'html': ''
        }
        
        prediction, confidence = email_classifier.predict(test_email)
        
        print(f"  Subject: {test_email['subject']}")
        print(f"  Prediction: {'PHISHING' if prediction == 1 else 'LEGITIMATE'}")
        print(f"  Confidence: {confidence:.2%}")
        print("  ✓ Email classifier working!")
        
        print("\n✓ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Testing failed: {e}")
        return False

def main():
    """Main setup function"""
    print("\n")
    print("=" * 70)
    print(" " * 15 + "PHISHING DETECTION SYSTEM - SETUP")
    print("=" * 70)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install dependencies first!")
        return
    
    # Step 2: Ask if user wants to train
    print("\n")
    response = input("Do you want to train the models now? (y/n): ").strip().lower()
    
    if response == 'y':
        if not run_training():
            print("\n⚠️  Training failed. Please check the errors above.")
            return
    else:
        print("\nSkipping training. You can train later by running: python src/train.py")
        return
    
    # Step 3: Test predictions
    print("\n")
    response = input("Do you want to test the predictions? (y/n): ").strip().lower()
    
    if response == 'y':
        test_predictions()
    
    # Final instructions
    print_header("SETUP COMPLETE!")
    print("\nYou can now use the system in multiple ways:")
    print("\n1. Web Application:")
    print("   python app.py")
    print("   Then open: http://localhost:5000")
    print("\n2. Command Line:")
    print("   cd src")
    print("   python predict.py")
    print("\n3. Python API:")
    print("   See examples in README.md")
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
