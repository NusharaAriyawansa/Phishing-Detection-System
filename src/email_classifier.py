"""
Email Phishing Classifier
Trains and predicts phishing emails using machine learning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib
import os
from typing import Tuple, Dict
from feature_extraction import EmailFeatureExtractor


class EmailPhishingClassifier:
    """Classifier for detecting phishing emails"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the classifier
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'logistic_regression')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = EmailFeatureExtractor()
        self.feature_names = None
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def prepare_features(self, emails: list) -> pd.DataFrame:
        """
        Extract features from list of emails
        
        Args:
            emails: List of email dictionaries with keys: 'subject', 'body', 'sender', 'html' (optional)
        """
        features_list = []
        
        for email in emails:
            features = self.feature_extractor.extract_features(email)
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        # Store feature names
        if self.feature_names is None:
            self.feature_names = df.columns.tolist()
        
        return df
    
    def train(self, emails: list, labels: list, test_size: float = 0.2) -> Dict:
        """
        Train the classifier
        
        Args:
            emails: List of email dictionaries
            labels: List of labels (1 for phishing, 0 for legitimate)
            test_size: Proportion of dataset to use for testing
            
        Returns:
            Dictionary containing training metrics
        """
        print("Extracting features from emails...")
        X = self.prepare_features(emails)
        y = np.array(labels)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Phishing emails: {sum(y)}, Legitimate emails: {len(y) - sum(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        print("Evaluating model...")
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        metrics = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'])
        }
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        print(f"\nTraining Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Cross-Validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        return metrics
    
    def predict(self, email: Dict[str, str]) -> Tuple[int, float]:
        """
        Predict if an email is phishing
        
        Args:
            email: Email dictionary with keys: 'subject', 'body', 'sender', 'html' (optional)
            
        Returns:
            Tuple of (prediction, confidence)
            prediction: 1 for phishing, 0 for legitimate
            confidence: probability of phishing
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first or load a saved model.")
        
        # Extract features
        features = self.feature_extractor.extract_features(email)
        X = pd.DataFrame([features])
        
        # Ensure feature order matches training
        if self.feature_names:
            X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        
        # Get probability
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = probabilities[1]  # Probability of phishing
        else:
            confidence = prediction  # For models without predict_proba
        
        return int(prediction), float(confidence)
    
    def predict_batch(self, emails: list) -> list:
        """
        Predict multiple emails
        
        Args:
            emails: List of email dictionaries to classify
            
        Returns:
            List of tuples (prediction, confidence)
        """
        results = []
        for email in emails:
            prediction, confidence = self.predict(email)
            results.append((prediction, confidence))
        
        return results
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scaler"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Sample email data
    sample_emails = [
        {
            'subject': 'Meeting Tomorrow',
            'body': 'Hi, just a reminder about our meeting tomorrow at 2 PM. See you then!',
            'sender': 'colleague@company.com',
            'html': ''
        },
        {
            'subject': 'URGENT: Verify Your Account Now!!!',
            'body': 'Dear customer, your account has been suspended due to unusual activity. Click here to verify your identity immediately: http://fake-bank.com/verify or your account will be permanently closed.',
            'sender': 'security@tempmail123.com',
            'html': '<html><body>Click <a href="http://fake-bank.com">here</a></body></html>'
        },
        {
            'subject': 'Weekly Report',
            'body': 'Please find attached the weekly report for review.',
            'sender': 'reports@company.com',
            'html': ''
        },
        {
            'subject': 'Congratulations! You won $1,000,000',
            'body': 'Dear winner, you have been selected to receive $1,000,000. Click below to claim your prize now! Limited time offer. Act now!',
            'sender': 'winner@lottery-prize.xyz',
            'html': '<form action="http://scam.com"><input type="submit" value="Claim Prize"></form>'
        },
        {
            'subject': 'Invoice for Services',
            'body': 'Thank you for your business. Please find the invoice attached.',
            'sender': 'billing@vendor.com',
            'html': ''
        },
        {
            'subject': 'Your payment has failed',
            'body': 'Dear user, your recent payment has failed. Update your billing information immediately to avoid service interruption. Click here: http://update-billing.tk',
            'sender': 'no-reply@service123.com',
            'html': ''
        },
        {
            'subject': 'Project Update',
            'body': 'The project is on track. Next milestone is scheduled for next week.',
            'sender': 'pm@company.com',
            'html': ''
        },
        {
            'subject': 'Confirm your identity - URGENT',
            'body': 'DEAR CUSTOMER, we have detected suspicious activity. Verify your account now: http://verify-account.ml/?id=12345. Your social security and credit card need confirmation.',
            'sender': 'security999@guerrillamail.com',
            'html': '<html><body style="display:none">Hidden tracking</body></html>'
        },
    ]
    
    # Labels: 0 = legitimate, 1 = phishing
    sample_labels = [0, 1, 0, 1, 0, 1, 0, 1]
    
    # Create and train classifier
    classifier = EmailPhishingClassifier(model_type='random_forest')
    
    print("Training Email Phishing Classifier...")
    print("=" * 60)
    
    metrics = classifier.train(sample_emails, sample_labels, test_size=0.25)
    
    # Test predictions
    print("\n" + "=" * 60)
    print("Testing Predictions:")
    print("=" * 60)
    
    test_emails = [
        {
            'subject': 'Lunch today?',
            'body': 'Hey, want to grab lunch today around noon?',
            'sender': 'friend@email.com',
            'html': ''
        },
        {
            'subject': 'ACTION REQUIRED: Suspended Account',
            'body': 'Your account will be closed unless you verify immediately. Click here: http://verify-now.tk',
            'sender': 'alert@tempmail.com',
            'html': ''
        },
    ]
    
    for i, email in enumerate(test_emails, 1):
        prediction, confidence = classifier.predict(email)
        label = "PHISHING" if prediction == 1 else "LEGITIMATE"
        print(f"\nEmail {i}:")
        print(f"Subject: {email['subject']}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
    
    # Save model
    model_path = "../models/email_classifier.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    classifier.save_model(model_path)
