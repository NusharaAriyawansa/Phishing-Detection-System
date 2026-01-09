"""
Feature Extraction Module for Phishing Detection System
Extracts features from URLs and Email content for classification
"""

import re
import urllib.parse
from typing import Dict, List
import tldextract
import numpy as np
from collections import Counter


class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""
    
    def __init__(self):
        self.suspicious_words = [
            'secure', 'account', 'update', 'login', 'ebay', 'paypal',
            'banking', 'confirm', 'verify', 'password', 'signin', 'webscr'
        ]
        
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz']
        self.brand_keywords = [
            'paypal', 'apple', 'google', 'microsoft', 'amazon', 'netflix',
            'facebook', 'bank', 'login', 'secure'
        ]
    
    def extract_features(self, url: str) -> Dict[str, float]:
        """Extract comprehensive features from a URL"""
        features = {}
        
        # Basic URL features
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_question_marks'] = url.count('?')
        features['num_equals'] = url.count('=')
        features['num_at'] = url.count('@')
        features['num_ampersand'] = url.count('&')
        features['num_exclamation'] = url.count('!')
        features['num_tilde'] = url.count('~')
        features['num_comma'] = url.count(',')
        features['num_plus'] = url.count('+')
        features['num_asterisk'] = url.count('*')
        features['num_hash'] = url.count('#')
        features['num_dollar'] = url.count('$')
        features['num_percent'] = url.count('%')
        
        # Check for IP address
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        features['has_ip'] = 1.0 if re.search(ip_pattern, url) else 0.0
        
        # Protocol features
        features['is_https'] = 1.0 if url.startswith('https://') else 0.0
        features['is_http'] = 1.0 if url.startswith('http://') else 0.0
        
        # Extract domain
        try:
            extracted = tldextract.extract(url)
            domain = extracted.domain
            subdomain = extracted.subdomain
            suffix = extracted.suffix
            
            features['domain_length'] = len(domain) if domain else 0
            features['subdomain_length'] = len(subdomain) if subdomain else 0
            features['num_subdomains'] = len(subdomain.split('.')) if subdomain else 0
            
            # Check suspicious TLD
            features['suspicious_tld'] = 1.0 if f'.{suffix}' in self.suspicious_tlds else 0.0
            
            # Check for suspicious words in domain
            suspicious_count = sum(1 for word in self.suspicious_words if word in url.lower())
            features['suspicious_word_count'] = suspicious_count

            # Homograph and brand lookalike signals
            canonical_domain = self._normalize_leetspeak(domain.lower()) if domain else ''
            host_for_match = '.'.join([part for part in [subdomain, domain] if part]).lower()
            canonical_host = self._normalize_leetspeak(host_for_match)
            features['has_digit_in_domain'] = 1.0 if any(c.isdigit() for c in domain) else 0.0
            features['homograph_brand'] = 1.0 if any(brand in canonical_host for brand in self.brand_keywords) else 0.0
            features['brand_edit_distance'] = self._min_brand_distance(canonical_domain)
            
        except:
            features['domain_length'] = 0
            features['subdomain_length'] = 0
            features['num_subdomains'] = 0
            features['suspicious_tld'] = 0
            features['suspicious_word_count'] = 0
            features['has_digit_in_domain'] = 0.0
            features['homograph_brand'] = 0.0
            features['brand_edit_distance'] = 1.0
        
        # Path features
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        query = parsed.query
        
        features['path_length'] = len(path)
        features['query_length'] = len(query)
        
        # Digit ratio
        digits = sum(c.isdigit() for c in url)
        features['digit_ratio'] = digits / len(url) if len(url) > 0 else 0
        
        # Letter ratio
        letters = sum(c.isalpha() for c in url)
        features['letter_ratio'] = letters / len(url) if len(url) > 0 else 0
        
        # Check for URL shortening
        shortening_services = ['bit.ly', 'goo.gl', 'tinyurl', 'ow.ly', 't.co', 'is.gd']
        features['is_shortened'] = 1.0 if any(service in url.lower() for service in shortening_services) else 0.0
        
        # Entropy of URL
        features['entropy'] = self._calculate_entropy(url)
        
        return features

    def _normalize_leetspeak(self, text: str) -> str:
        """Normalize common digit/character substitutions used in homographs."""
        table = str.maketrans({
            '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's', '7': 't',
            '@': 'a', '$': 's'
        })
        return text.translate(table)

    def _levenshtein(self, a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        previous_row = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            current_row = [i]
            for j, cb in enumerate(b, 1):
                insertions = previous_row[j] + 1
                deletions = current_row[j - 1] + 1
                substitutions = previous_row[j - 1] + (ca != cb)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def _min_brand_distance(self, text: str) -> float:
        if not text:
            return 1.0
        distances = []
        for brand in self.brand_keywords:
            dist = self._levenshtein(text, brand) / max(len(brand), 1)
            distances.append(dist)
        return float(min(distances)) if distances else 1.0
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        counter = Counter(text)
        length = len(text)
        entropy = 0.0
        
        for count in counter.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy


class EmailFeatureExtractor:
    """Extract features from emails for phishing detection"""
    
    def __init__(self):
        self.phishing_keywords = [
            'urgent', 'verify', 'suspended', 'click here', 'confirm', 
            'account', 'update', 'security', 'password', 'credit card',
            'social security', 'expire', 'winner', 'congratulations',
            'prize', 'inheritance', 'lottery', 'claim', 'limited time',
            'act now', 'dear customer', 'dear user', 'verify your account',
            'suspended account', 'unusual activity', 'confirm identity',
            'billing information', 'payment failed', 'click below'
        ]
        
        self.suspicious_domains = [
            'tempmail', 'guerrillamail', 'mailinator', '10minutemail'
        ]
    
    def extract_features(self, email_data: Dict[str, str]) -> Dict[str, float]:
        """
        Extract features from email
        email_data should contain: 'subject', 'body', 'sender', 'html' (optional)
        """
        features = {}
        
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        sender = email_data.get('sender', '')
        html = email_data.get('html', '')
        
        # Subject features
        features['subject_length'] = len(subject)
        features['subject_has_urgency'] = 1.0 if any(word in subject.lower() for word in ['urgent', 'immediate', 'act now']) else 0.0
        features['subject_all_caps'] = 1.0 if subject.isupper() and len(subject) > 5 else 0.0
        features['subject_exclamation'] = subject.count('!')
        
        # Body features
        features['body_length'] = len(body)
        
        # Phishing keyword count
        body_lower = body.lower()
        keyword_count = sum(1 for keyword in self.phishing_keywords if keyword in body_lower)
        features['phishing_keyword_count'] = keyword_count
        
        # URL count in body
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls_in_body = re.findall(url_pattern, body)
        features['url_count'] = len(urls_in_body)
        
        # Check for mismatched URLs (displayed text vs actual link)
        features['has_mismatched_url'] = 0.0  # Would need HTML parsing for accurate detection
        
        # Sender features
        features['sender_length'] = len(sender)
        
        # Check for suspicious sender domain
        sender_suspicious = 1.0 if any(domain in sender.lower() for domain in self.suspicious_domains) else 0.0
        features['sender_suspicious'] = sender_suspicious
        
        # Check if sender has numbers
        features['sender_has_numbers'] = 1.0 if any(c.isdigit() for c in sender) else 0.0
        
        # HTML features (if HTML content available)
        if html:
            features['has_html'] = 1.0
            features['html_length'] = len(html)
            
            # Check for hidden content
            features['has_hidden_content'] = 1.0 if 'display:none' in html or 'visibility:hidden' in html else 0.0
            
            # Check for external images
            features['external_image_count'] = html.count('<img') + html.count('<IMG')
            
            # Check for forms
            features['has_form'] = 1.0 if '<form' in html.lower() else 0.0
        else:
            features['has_html'] = 0.0
            features['html_length'] = 0
            features['has_hidden_content'] = 0.0
            features['external_image_count'] = 0
            features['has_form'] = 0.0
        
        # Punctuation analysis
        features['exclamation_count'] = body.count('!')
        features['question_count'] = body.count('?')
        
        # Uppercase ratio
        uppercase = sum(1 for c in body if c.isupper())
        features['uppercase_ratio'] = uppercase / len(body) if len(body) > 0 else 0
        
        # Check for generic greeting
        generic_greetings = ['dear customer', 'dear user', 'dear member', 'dear sir/madam']
        features['has_generic_greeting'] = 1.0 if any(greeting in body.lower() for greeting in generic_greetings) else 0.0
        
        # Entropy
        features['body_entropy'] = self._calculate_entropy(body)
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        counter = Counter(text)
        length = len(text)
        entropy = 0.0
        
        for count in counter.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy


# Example usage
if __name__ == "__main__":
    # Test URL feature extraction
    url_extractor = URLFeatureExtractor()
    
    test_url = "http://secure-paypal-login.tk/update?user=12345"
    url_features = url_extractor.extract_features(test_url)
    
    print("URL Features:")
    for key, value in url_features.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    # Test email feature extraction
    email_extractor = EmailFeatureExtractor()
    
    test_email = {
        'subject': 'URGENT: Verify Your Account Now!',
        'body': 'Dear customer, your account has been suspended due to unusual activity. Click here to verify your identity immediately: http://fake-bank.com/verify',
        'sender': 'security@temp-mail123.com',
        'html': '<html><body>Click <a href="http://fake-bank.com">here</a></body></html>'
    }
    
    email_features = email_extractor.extract_features(test_email)
    
    print("Email Features:")
    for key, value in email_features.items():
        print(f"  {key}: {value}")
