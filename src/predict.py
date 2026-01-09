"""
Prediction Script for Phishing Detection System
Makes predictions using trained models
"""

import sys
import os
import argparse
from url_classifier import URLPhishingClassifier
from email_classifier import EmailPhishingClassifier


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def predict_url(url: str, model_path: str = None):
    """
    Predict if a URL is phishing
    
    Args:
        url: URL to check
        model_path: Path to saved model (optional)
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, 'url_classifier.pkl')
    
    # Load classifier
    classifier = URLPhishingClassifier()
    
    try:
        classifier.load_model(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running: python train.py")
        return
    
    # Make prediction
    prediction, confidence = classifier.predict(url)
    
    # Display results
    print("\n" + "=" * 60)
    print("URL PHISHING DETECTION RESULTS")
    print("=" * 60)
    print(f"\nURL: {url}")
    print(f"\nPrediction: {'PHISHING ⚠️' if prediction == 1 else 'LEGITIMATE ✓'}")
    print(f"Confidence: {confidence:.2%}")
    
    if prediction == 1:
        print("\n⚠️  WARNING: This URL appears to be a phishing attempt!")
        print("Do NOT click on this link or enter any personal information.")
    else:
        print("\n✓ This URL appears to be legitimate.")
    
    print("=" * 60 + "\n")


def predict_email(subject: str, body: str, sender: str, html: str = "", model_path: str = None):
    """
    Predict if an email is phishing
    
    Args:
        subject: Email subject
        body: Email body text
        sender: Sender email address
        html: HTML content (optional)
        model_path: Path to saved model (optional)
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, 'email_classifier.pkl')
    
    # Load classifier
    classifier = EmailPhishingClassifier()
    
    try:
        classifier.load_model(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running: python train.py")
        return
    
    # Prepare email data
    email_data = {
        'subject': subject,
        'body': body,
        'sender': sender,
        'html': html
    }
    
    # Make prediction
    prediction, confidence = classifier.predict(email_data)
    
    # Display results
    print("\n" + "=" * 60)
    print("EMAIL PHISHING DETECTION RESULTS")
    print("=" * 60)
    print(f"\nFrom: {sender}")
    print(f"Subject: {subject}")
    print(f"\nBody Preview: {body[:100]}{'...' if len(body) > 100 else ''}")
    print(f"\nPrediction: {'PHISHING ⚠️' if prediction == 1 else 'LEGITIMATE ✓'}")
    print(f"Confidence: {confidence:.2%}")
    
    if prediction == 1:
        print("\n⚠️  WARNING: This email appears to be a phishing attempt!")
        print("Do NOT click any links or provide personal information.")
        print("Common phishing indicators:")
        print("  - Urgent or threatening language")
        print("  - Requests for personal/financial information")
        print("  - Suspicious sender address")
        print("  - Generic greetings (e.g., 'Dear Customer')")
    else:
        print("\n✓ This email appears to be legitimate.")
    
    print("=" * 60 + "\n")


def interactive_url_mode():
    """Interactive mode for checking URLs"""
    print("\n" + "=" * 60)
    print("URL PHISHING DETECTION - INTERACTIVE MODE")
    print("=" * 60)
    print("\nEnter URLs to check (type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        url = input("\nEnter URL: ").strip()
        
        if url.lower() in ['quit', 'exit', 'q']:
            print("\nExiting...")
            break
        
        if not url:
            print("Please enter a valid URL")
            continue
        
        predict_url(url)


def interactive_email_mode():
    """Interactive mode for checking emails"""
    print("\n" + "=" * 60)
    print("EMAIL PHISHING DETECTION - INTERACTIVE MODE")
    print("=" * 60)
    print("\nEnter email details (type 'quit' at any prompt to exit)")
    print("-" * 60)
    
    while True:
        sender = input("\nSender email: ").strip()
        if sender.lower() in ['quit', 'exit', 'q']:
            print("\nExiting...")
            break
        
        subject = input("Subject: ").strip()
        if subject.lower() in ['quit', 'exit', 'q']:
            print("\nExiting...")
            break
        
        print("Body (press Enter twice when done):")
        body_lines = []
        while True:
            line = input()
            if line:
                body_lines.append(line)
            else:
                break
        body = '\n'.join(body_lines)
        
        if not sender or not subject or not body:
            print("Please provide all required fields")
            continue
        
        predict_email(subject, body, sender)


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Phishing Detection System - Prediction')
    parser.add_argument('--mode', choices=['url', 'email'], help='Detection mode (url or email)')
    parser.add_argument('--url', help='URL to check')
    parser.add_argument('--sender', help='Email sender address')
    parser.add_argument('--subject', help='Email subject')
    parser.add_argument('--body', help='Email body text')
    parser.add_argument('--model', help='Path to model file')
    
    args = parser.parse_args()
    
    # If no arguments provided, show menu
    if len(sys.argv) == 1:
        print("\n" + "=" * 60)
        print(" " * 15 + "PHISHING DETECTION SYSTEM")
        print("=" * 60)
        print("\nSelect detection mode:")
        print("  1. URL Phishing Detection")
        print("  2. Email Phishing Detection")
        print("  3. Exit")
        print("-" * 60)
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            interactive_url_mode()
        elif choice == '2':
            interactive_email_mode()
        elif choice == '3':
            print("\nExiting...")
        else:
            print("\nInvalid choice!")
    
    # URL prediction
    elif args.mode == 'url' and args.url:
        predict_url(args.url, args.model)
    
    # Email prediction
    elif args.mode == 'email' and args.sender and args.subject and args.body:
        predict_email(args.subject, args.body, args.sender, model_path=args.model)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
