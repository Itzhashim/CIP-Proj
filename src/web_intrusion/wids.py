"""
Web Intrusion Detection System (WIDS)

Simulates web log data and detects anomalies using Isolation Forest.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42


class WebIntrusionDetector:
    """Isolation Forest-based anomaly detection for web intrusions"""
    
    def __init__(self, base_dir):
        """
        Initialize the detector
        
        Args:
            base_dir: Base project directory path
        """
        self.base_dir = base_dir
        self.model_dir = os.path.join(base_dir, 'models')
        self.model = None
        self.scaler = StandardScaler()
        
        # Ensure models directory exists
        os.makedirs(self.model_dir, exist_ok=True)
    
    def generate_web_log_data(self, n_samples=10000):
        """
        Generate simulated web log data based on network traffic features
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame: Simulated web log data
        """
        print("\n" + "="*80)
        print("MODULE 2: WEB INTRUSION DETECTION (ANOMALY)")
        print("="*80)
        
        print(f"\n[SIMULATION] Generating {n_samples} synthetic web log entries...")
        
        np.random.seed(RANDOM_STATE)
        
        # Simulate normal traffic (80%)
        n_normal = int(n_samples * 0.8)
        normal_data = {
            'request_size': np.random.normal(500, 100, n_normal),
            'response_size': np.random.normal(2000, 500, n_normal),
            'response_time': np.random.gamma(2, 50, n_normal),
            'requests_per_min': np.random.poisson(10, n_normal),
            'error_rate': np.random.beta(1, 50, n_normal),
            'unique_pages': np.random.poisson(5, n_normal),
        }
        
        # Simulate anomalous traffic (20%)
        n_anomaly = n_samples - n_normal
        anomaly_data = {
            'request_size': np.random.normal(5000, 1000, n_anomaly),  # Larger requests
            'response_size': np.random.normal(100, 50, n_anomaly),    # Smaller responses
            'response_time': np.random.gamma(5, 200, n_anomaly),      # Slower
            'requests_per_min': np.random.poisson(100, n_anomaly),    # More frequent
            'error_rate': np.random.beta(10, 20, n_anomaly),          # More errors
            'unique_pages': np.random.poisson(1, n_anomaly),          # Fewer unique pages
        }
        
        # Combine
        df_normal = pd.DataFrame(normal_data)
        df_anomaly = pd.DataFrame(anomaly_data)
        df = pd.concat([df_normal, df_anomaly], axis=0, ignore_index=True)
        
        # Shuffle
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        print(f"[SIMULATION] Generated data shape: {df.shape}")
        print(f"[SIMULATION] Features: {list(df.columns)}")
        
        return df
    
    def train(self, X):
        """
        Train Isolation Forest on web log data
        
        Args:
            X: Feature DataFrame
            
        Returns:
            dict: Training results including anomaly count
        """
        print(f"\n[TRAINING] Training Isolation Forest...")
        print(f"[TRAINING] Input shape: {X.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=0.2,  # Expect 20% anomalies
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        
        predictions = self.model.fit_predict(X_scaled)
        
        # -1 for anomalies, 1 for normal
        n_anomalies = sum(predictions == -1)
        n_normal = sum(predictions == 1)
        
        print(f"\n[RESULTS] Detection Results:")
        print(f"  Normal traffic: {n_normal} ({100*n_normal/len(predictions):.1f}%)")
        print(f"  Anomalies detected: {n_anomalies} ({100*n_anomalies/len(predictions):.1f}%)")
        
        # Save model
        model_path = os.path.join(self.model_dir, 'wids_iforest.pkl')
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, model_path)
        
        print(f"\n[SAVED] Model saved to: {model_path}")
        print("✓ Web Intrusion Detection completed successfully")
        
        return {
            'success': True,
            'model': self.model,
            'anomaly_count': n_anomalies,
            'total_samples': len(predictions),
            'model_path': model_path
        }
    
    def run(self, n_samples=10000):
        """
        Run the complete web intrusion detection pipeline
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            dict: Training results
        """
        X = self.generate_web_log_data(n_samples)
        results = self.train(X)
        return results
    
    def predict(self, X_sample):
        """
        Make prediction on a sample
        
        Args:
            X_sample: Feature array
            
        Returns:
            array: Predictions (-1 for anomaly, 1 for normal)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call run() or train() first.")
        
        X_scaled = self.scaler.transform(X_sample)
        prediction = self.model.predict(X_scaled)
        return prediction


def train_wids():
    """
    Main function to train web intrusion detection system
    
    Returns:
        dict: Training results including anomaly count
    """
    # Determine base directory (project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    
    # Initialize and train detector
    detector = WebIntrusionDetector(base_dir)
    results = detector.run()
    
    return results


if __name__ == "__main__":
    # Allow module to be run standalone for testing
    results = train_wids()
    if results['success']:
        print(f"\nDetected {results['anomaly_count']} anomalies out of {results['total_samples']} samples")
    else:
        print("\nTraining failed!")
