"""
Network Traffic Classification Module

Handles loading, preprocessing, and classification of network traffic data
from multiple datasets (NSL-KDD, CIC-IDS-2017, UNSW-NB15).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuration
SAMPLE_SIZE = 50000
RANDOM_STATE = 42


class NetworkTrafficClassifier:
    """Random Forest-based network traffic classifier"""
    
    def __init__(self, base_dir):
        """
        Initialize the classifier
        
        Args:
            base_dir: Base project directory path
        """
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.model_dir = os.path.join(base_dir, 'models')
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
        # Ensure models directory exists
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_and_preprocess_dataset(self, file_path, dataset_name):
        """
        Load and preprocess a single dataset
        
        Args:
            file_path: Path to CSV file
            dataset_name: Name for logging
            
        Returns:
            tuple: (X, y) features and labels
        """
        print(f"\n[{dataset_name}] Loading dataset from {os.path.basename(file_path)}...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"[{dataset_name}] Original shape: {df.shape}")
            
            # Sample if needed
            if len(df) > SAMPLE_SIZE:
                df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
                print(f"[{dataset_name}] Sampled to: {df.shape}")
            
            # Identify label column (usually last column or contains 'label'/'class')
            label_col = None
            for col in df.columns:
                if 'label' in col.lower() or 'class' in col.lower() or col == df.columns[-1]:
                    label_col = col
                    break
            
            if label_col is None:
                label_col = df.columns[-1]
            
            print(f"[{dataset_name}] Label column: {label_col}")
            
            # Separate features and labels
            X = df.drop(columns=[label_col])
            y = df[label_col]
            
            # Convert labels to binary (normal=0, attack=1)
            if y.dtype == 'object':
                y_binary = y.apply(lambda x: 0 if 'normal' in str(x).lower() else 1)
            else:
                y_binary = (y != 0).astype(int)
            
            print(f"[{dataset_name}] Class distribution:")
            print(f"  Normal: {sum(y_binary == 0)} ({100*sum(y_binary == 0)/len(y_binary):.1f}%)")
            print(f"  Attack: {sum(y_binary == 1)} ({100*sum(y_binary == 1)/len(y_binary):.1f}%)")
            
            # Handle categorical features
            for col in X.columns:
                if X[col].dtype == 'object':
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                    else:
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
            
            # Convert to numeric and handle any remaining issues
            X = X.apply(pd.to_numeric, errors='coerce')
            X = X.fillna(0)
            
            return X, y_binary
            
        except Exception as e:
            print(f"[{dataset_name}] ERROR: {str(e)}")
            return None, None
    
    def train_on_multiple_datasets(self):
        """
        Train classifier on multiple datasets
        
        Returns:
            dict: Training results including model and accuracy
        """
        print("\n" + "="*80)
        print("MODULE 1: NETWORK TRAFFIC CLASSIFICATION")
        print("="*80)
        
        datasets = [
            ('NSL_KDD.csv', 'NSL-KDD'),
            ('CIC_IDS_2017.csv', 'CIC-IDS-2017'),
            ('UNSW_NB15.csv', 'UNSW-NB15')
        ]
        
        X_combined = []
        y_combined = []
        
        # Load all datasets
        for filename, name in datasets:
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                X, y = self.load_and_preprocess_dataset(file_path, name)
                if X is not None:
                    X_combined.append(X)
                    y_combined.append(y)
            else:
                print(f"[{name}] WARNING: File not found at {file_path}, skipping...")
        
        if not X_combined:
            print("\nERROR: No datasets could be loaded!")
            return {'success': False, 'model': None, 'accuracy': 0.0}
        
        # Combine datasets
        print("\n[COMBINING] Merging all datasets...")
        
        # Get common columns
        common_cols = set(X_combined[0].columns)
        for X in X_combined[1:]:
            common_cols = common_cols.intersection(set(X.columns))
        common_cols = list(common_cols)
        
        print(f"[COMBINING] Using {len(common_cols)} common features")
        
        # Use only common columns
        X_combined = [X[common_cols] for X in X_combined]
        
        # Concatenate
        X_all = pd.concat(X_combined, axis=0, ignore_index=True)
        y_all = pd.concat(y_combined, axis=0, ignore_index=True)
        
        print(f"[COMBINING] Final combined shape: {X_all.shape}")
        print(f"[COMBINING] Final class distribution:")
        print(f"  Normal: {sum(y_all == 0)} ({100*sum(y_all == 0)/len(y_all):.1f}%)")
        print(f"  Attack: {sum(y_all == 1)} ({100*sum(y_all == 1)/len(y_all):.1f}%)")
        
        # Store feature columns
        self.feature_columns = common_cols
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
        )
        
        print(f"\n[TRAINING] Training set: {X_train.shape}")
        print(f"[TRAINING] Test set: {X_test.shape}")
        
        # Scale features
        print("[TRAINING] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("[TRAINING] Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        print("\n[EVALUATION] Making predictions...")
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n[RESULTS] Accuracy: {accuracy:.4f}")
        
        print("\n[RESULTS] Classification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=['Normal', 'Attack'],
            digits=4
        ))
        
        # Save model
        model_path = os.path.join(self.model_dir, 'network_rf.pkl')
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }, model_path)
        
        print(f"\n[SAVED] Model saved to: {model_path}")
        print("✓ Network Traffic Classification completed successfully")
        
        return {
            'success': True,
            'model': self.model,
            'accuracy': accuracy,
            'model_path': model_path
        }
    
    def predict(self, X_sample):
        """
        Make prediction on a sample
        
        Args:
            X_sample: Feature array
            
        Returns:
            array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_on_multiple_datasets() first.")
        
        X_scaled = self.scaler.transform(X_sample)
        prediction = self.model.predict(X_scaled)
        return prediction


def train_network_classifier():
    """
    Main function to train network traffic classifier
    
    Returns:
        dict: Training results including model and accuracy
    """
    # Determine base directory (project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    
    # Initialize and train classifier
    classifier = NetworkTrafficClassifier(base_dir)
    results = classifier.train_on_multiple_datasets()
    
    return results


if __name__ == "__main__":
    # Allow module to be run standalone for testing
    results = train_network_classifier()
    if results['success']:
        print(f"\nTraining completed with accuracy: {results['accuracy']:.4f}")
    else:
        print("\nTraining failed!")
