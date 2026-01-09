"""
URL Phishing Classifier
Trains and predicts phishing URLs using machine learning
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
from feature_extraction import URLFeatureExtractor


class URLPhishingClassifier:
    """Classifier for detecting phishing URLs"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the classifier
        
        Args:
            model_type: Type of model to use ('random_forest', 'gradient_boosting', 'logistic_regression')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = URLFeatureExtractor()
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
    
    def prepare_features(self, urls: list) -> pd.DataFrame:
        """Extract features from list of URLs"""
        features_list = []
        
        for url in urls:
            features = self.feature_extractor.extract_features(url)
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        
        # Store feature names
        if self.feature_names is None:
            self.feature_names = df.columns.tolist()
        
        return df
    
    def train(self, urls: list, labels: list, test_size: float = 0.2) -> Dict:
        """
        Train the classifier
        
        Args:
            urls: List of URLs
            labels: List of labels (1 for phishing, 0 for legitimate)
            test_size: Proportion of dataset to use for testing
            
        Returns:
            Dictionary containing training metrics
        """
        print("Extracting features from URLs...")
        X = self.prepare_features(urls)
        y = np.array(labels)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Phishing URLs: {sum(y)}, Legitimate URLs: {len(y) - sum(y)}")
        
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
    
    def predict(self, url: str) -> Tuple[int, float]:
        """
        Predict if a URL is phishing
        
        Args:
            url: URL to classify
            
        Returns:
            Tuple of (prediction, confidence)
            prediction: 1 for phishing, 0 for legitimate
            confidence: probability of phishing
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first or load a saved model.")
        
        # Extract features
        features = self.feature_extractor.extract_features(url)
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
    
    def predict_batch(self, urls: list) -> list:
        """
        Predict multiple URLs
        
        Args:
            urls: List of URLs to classify
            
        Returns:
            List of tuples (prediction, confidence)
        """
        results = []
        for url in urls:
            prediction, confidence = self.predict(url)
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
        
        self.model = model_data.get('model')
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data.get('feature_names')
        self.model_type = model_data.get('model_type')
        
        # Validate that scaler is fitted
        if self.scaler is None or not hasattr(self.scaler, 'mean_'):
            raise ValueError("Loaded scaler is not fitted. The model file may be corrupted. Please retrain the model.")
        
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    # Example data (in practice, you would load this from a dataset)
    sample_urls = [
        "https://www.google.com",
        "https://www.microsoft.com",
        "http://secure-paypal-verify.tk/login",
        "http://192.168.1.1/admin",
        "https://www.amazon.com/products",
        "http://bit.ly/2xYz3aB",
        "http://verify-account-now.ml/update?user=12345",
        "https://github.com/username/repo",
        "http://free-prize-winner.xyz/claim",
        "https://www.wikipedia.org",
    ]
    
    # Labels: 0 = legitimate, 1 = phishing
    sample_labels = [0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    
    # Create and train classifier
    classifier = URLPhishingClassifier(model_type='random_forest')
    
    print("Training URL Phishing Classifier...")
    print("=" * 60)
    
    metrics = classifier.train(sample_urls, sample_labels, test_size=0.2)
    
    # Test predictions
    print("\n" + "=" * 60)
    print("Testing Predictions:")
    print("=" * 60)
    
    test_urls = [
        "https://www.example.com",
        "http://secure-login-verify.tk/update",
    ]
    
    for url in test_urls:
        prediction, confidence = classifier.predict(url)
        label = "PHISHING" if prediction == 1 else "LEGITIMATE"
        print(f"\nURL: {url}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")
    
    # Save model
    model_path = "../models/url_classifier.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    classifier.save_model(model_path)
