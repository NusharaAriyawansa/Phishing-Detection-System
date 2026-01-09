"""
Training Script for Phishing Detection System
Trains both URL and Email classifiers using datasets from the data folder

Default datasets:
    - data/sample_urls.csv (URL dataset)
    - data/sample_emails.csv (Email dataset)

Supported dataset formats
-------------------------
URL CSV:
    - Required columns: `url`, `Type` (case-insensitive)
    - `Type` values treated as phishing if they are one of
      {"phishing", "spam", "malicious", "1", "true", "yes"}; otherwise legitimate

Email CSV:
    - Required columns (case-insensitive):
        * Subject (e.g., "Subject")
        * Body / Message text (e.g., "Message", "Body", "Text")
        * Label column (e.g., "Spam/Ham", "Label", "Class")
    - Optional: Sender column (e.g., "From", "Sender")
    - Label is phishing if in {"spam", "phishing", "1", "true", "yes"}

Usage:
    python train.py                              # Uses data folder files
    python train.py --url-dataset path/to/urls.csv --email-dataset path/to/emails.csv
"""

import sys
import os
import argparse
from typing import Optional
import pandas as pd
import numpy as np
from url_classifier import URLPhishingClassifier
from email_classifier import EmailPhishingClassifier


PHISHING_LABEL_VALUES = {"phishing", "spam", "malicious", "1", "true", "yes"}
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def _pick_column(df: pd.DataFrame, candidates):
    """Pick the first matching column name from a list of candidates (case-insensitive)."""
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def load_url_csv(csv_path: str):
    """Load URL dataset from CSV with columns [url, Type]."""
    df = pd.read_csv(csv_path)

    url_col = _pick_column(df, ["url", "urls", "link", "links"])
    label_col = _pick_column(df, ["type", "label", "class", "target", "spam/ham", "spam"])

    if not url_col or not label_col:
        raise ValueError(
            f"CSV '{csv_path}' must contain URL column (e.g., 'url') and label column (e.g., 'Type')."
        )

    df = df[[url_col, label_col]].dropna()
    df[url_col] = df[url_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    urls = df[url_col].tolist()
    labels = [1 if v in PHISHING_LABEL_VALUES else 0 for v in df[label_col]]

    return urls, labels


def load_email_csv(csv_path: str):
    """Load Email dataset from CSV with columns [Subject, Message/Body/Text, Spam/Ham]."""
    df = pd.read_csv(csv_path)

    subject_col = _pick_column(df, ["subject"])
    body_col = _pick_column(df, ["message", "body", "text", "content"])
    label_col = _pick_column(df, ["spam/ham", "label", "class", "target", "spam"])
    sender_col = _pick_column(df, ["from", "sender", "email", "address", "from_address"])

    if not subject_col or not body_col or not label_col:
        raise ValueError(
            f"CSV '{csv_path}' must contain subject, body/message, and label columns."
        )

    df = df[[subject_col, body_col, label_col] + ([sender_col] if sender_col else [])].dropna()

    df[subject_col] = df[subject_col].astype(str)
    df[body_col] = df[body_col].astype(str)
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    if sender_col:
        df[sender_col] = df[sender_col].astype(str)

    emails = []
    labels = []

    for _, row in df.iterrows():
        email = {
            'subject': row[subject_col],
            'body': row[body_col],
            'sender': row[sender_col] if sender_col else 'unknown@example.com',
            'html': ''
        }
        label_val = str(row[label_col]).lower()
        label = 1 if label_val in PHISHING_LABEL_VALUES else 0
        emails.append(email)
        labels.append(label)

    return emails, labels


def train_url_classifier(url_csv: Optional[str] = None):
    """Train the URL phishing classifier using data from the data folder.

    Args:
        url_csv: Optional path to URL CSV dataset. If not provided, uses data/sample_urls.csv
    """
    print("\n" + "=" * 70)
    print("TRAINING URL PHISHING CLASSIFIER")
    print("=" * 70)

    if url_csv:
        print(f"\nLoading URL dataset from: {url_csv}")
        urls, labels = load_url_csv(url_csv)
    else:
        # Use data folder sample file
        default_csv = os.path.join(PROJECT_ROOT, "data", "sample_urls.csv")
        if not os.path.exists(default_csv):
            raise FileNotFoundError(
                f"URL dataset not found at {default_csv}. "
                "Please ensure data/sample_urls.csv exists or provide a custom dataset path."
            )
        print(f"\nLoading URL dataset from: {default_csv}")
        urls, labels = load_url_csv(default_csv)

    print(f"\nDataset size: {len(urls)} URLs")
    print(f"Legitimate: {labels.count(0)}, Phishing: {labels.count(1)}")

    # Train classifier
    classifier = URLPhishingClassifier(model_type='random_forest')
    metrics = classifier.train(urls, labels, test_size=0.25)

    # Save model
    model_path = os.path.join(MODELS_DIR, 'url_classifier.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    classifier.save_model(model_path)

    return classifier, metrics


def train_email_classifier(email_csv: Optional[str] = None):
    """Train the email phishing classifier using data from the data folder.

    Args:
        email_csv: Optional path to email CSV dataset. If not provided, uses data/sample_emails.csv
    """
    print("\n" + "=" * 70)
    print("TRAINING EMAIL PHISHING CLASSIFIER")
    print("=" * 70)

    if email_csv:
        print(f"\nLoading email dataset from: {email_csv}")
        emails, labels = load_email_csv(email_csv)
    else:
        # Use data folder sample file
        default_csv = os.path.join(PROJECT_ROOT, "data", "sample_emails.csv")
        if not os.path.exists(default_csv):
            raise FileNotFoundError(
                f"Email dataset not found at {default_csv}. "
                "Please ensure data/sample_emails.csv exists or provide a custom dataset path."
            )
        print(f"\nLoading email dataset from: {default_csv}")
        emails, labels = load_email_csv(default_csv)

    print(f"\nDataset size: {len(emails)} emails")
    print(f"Legitimate: {labels.count(0)}, Phishing: {labels.count(1)}")

    # Train classifier
    classifier = EmailPhishingClassifier(model_type='random_forest')
    metrics = classifier.train(emails, labels, test_size=0.25)

    # Save model
    model_path = os.path.join(MODELS_DIR, 'email_classifier.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    classifier.save_model(model_path)

    return classifier, metrics


def main():
    """Main training function"""
    args = parse_args()
    print("\n")
    print("=" * 70)
    print(" " * 15 + "PHISHING DETECTION SYSTEM - TRAINING")
    print("=" * 70)
    
    # Train URL classifier
    url_classifier, url_metrics = train_url_classifier(args.url_dataset)
    
    # Train Email classifier
    email_classifier, email_metrics = train_email_classifier(args.email_dataset)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    print("\nURL Classifier:")
    print(f"  Test Accuracy: {url_metrics['test_accuracy']:.2%}")
    print(f"  Precision: {url_metrics['precision']:.2%}")
    print(f"  Recall: {url_metrics['recall']:.2%}")
    print(f"  F1-Score: {url_metrics['f1_score']:.2%}")
    
    print("\nEmail Classifier:")
    print(f"  Test Accuracy: {email_metrics['test_accuracy']:.2%}")
    print(f"  Precision: {email_metrics['precision']:.2%}")
    print(f"  Recall: {email_metrics['recall']:.2%}")
    print(f"  F1-Score: {email_metrics['f1_score']:.2%}")
    
    print("\n" + "=" * 70)
    print("Models saved successfully in 'models/' directory")
    print("=" * 70 + "\n")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train phishing detection models")
    parser.add_argument(
        "--url-dataset",
        help="Path to URL CSV dataset (columns: url, Type)",
    )
    parser.add_argument(
        "--email-dataset",
        help="Path to Email CSV dataset (columns: Subject, Message/Body, Spam/Ham)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
