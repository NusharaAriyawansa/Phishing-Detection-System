"""
Phishing Detection System
A machine learning-based system for detecting phishing URLs and emails
"""

__version__ = '1.0.0'
__author__ = 'Nushara Ariyawansa'

from .feature_extraction import URLFeatureExtractor, EmailFeatureExtractor
from .url_classifier import URLPhishingClassifier
from .email_classifier import EmailPhishingClassifier

__all__ = [
    'URLFeatureExtractor',
    'EmailFeatureExtractor',
    'URLPhishingClassifier',
    'EmailPhishingClassifier',
]
